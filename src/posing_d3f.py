import torch.nn as nn
import torch
import numpy as np
import trimesh
import os
import pickle
from torch.autograd.functional import jacobian
import torch.nn.functional as F
from termcolor import colored
import time
import display_utils
import igl
from datetime import datetime,timezone
from PoissonSystem import poisson_system_matrices_from_mesh
from misc import shape_normalization_transforms_pytorch,get_current_time,convert_to_float_tensors,convert_to_float_tensors_from_numpy,convert_to_long_tensors_from_numpy,convert_to_long_tensors
import csv
from torch.autograd import Variable
import random
random.seed(10)
from torch.utils.data import DataLoader
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
from amass_utils import get_sequence,get_clips,get_time_for_clip,get_time_for_jacs,get_atomic_clips,get_pose_of_shape,split_by_indices,get_atomic_clips_by_joints
from timeit import default_timer as timer
import logging
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from math import ceil
from torch.optim.lr_scheduler import CosineAnnealingLR
from misc import compute_errors,align_feet,compute_soft,soft_displacements
from posing_network import GetNormals
from util_networks import CustomPositionalEncoding
import math
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_euler_angles,euler_angles_to_matrix

class D3FPosingTrainer():
    def __init__(self,posing_func,pointnet_func,seq_dataset,batch,frame_length,logdir,test_dataset=None):
        self.seq_dataset = seq_dataset
        self.test_seq_dataset = test_dataset
        self.num_seqs = len(seq_dataset)
        self.batch = batch
        self.logdir = logdir
        self.posing_func = posing_func
        self.pointnet_func = pointnet_func
        self.frame_length = frame_length
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.poisson_solver = None
        self.test_poisson_solver = None
        self.normals = GetNormals()
        self.positional_encoding = CustomPositionalEncoding(6)

    def preload(self,pose_weights,feature_weights,weight_path):
        #self.posing_func.load_state_dict(torch.load(pose_weights))
        #self.pointnet_func.load_state_dict(torch.load(feature_weights))
        self.posing_func.load_state_dict(torch.load(weight_path+'_posing'))
        self.pointnet_func.load_state_dict(torch.load(weight_path+'_ode_features'))

    def overwrite_networks(self,posing_func,pointnet_func):
        self.posing_func = posing_func
        self.pointnet_func = pointnet_func

    def batch_backprop(self,optimizer,batch_seq_loss,batch_jac_loss):
        loss = batch_seq_loss + 0.1*batch_jac_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.pointnet_func.zero_grad()
        self.posing_func.zero_grad()
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0

        return batch_seq_loss,batch_jac_loss

    def add_alignment_transform(self,seq,rots,trans):
        offset = torch.mean(seq,dim=1,keepdim=True)
        x90 = np.array([math.radians(-90),0,0])
        x90 = torch.from_numpy(x90).float().cuda()
        rot_mat = axis_angle_to_matrix(x90)

        z180 = np.array([0,0,math.radians(180)])
        z180 = torch.from_numpy(z180).float().cuda()
        rot_mat_z = axis_angle_to_matrix(z180)
        #rot_mat = torch.matmul(rot_mat,rot_mat_z)

        # comment out for "walk"
        #z90 = np.array([0,0,math.radians(-90)])
        #z90 = torch.from_numpy(z90).float().cuda()
        #rot_mat_z = axis_angle_to_matrix(z90)
        #rot_mat = torch.matmul(rot_mat,rot_mat_z)

        out_seq = seq - offset
        out_seq = torch.matmul(out_seq,rot_mat)
        out_seq += offset
        return out_seq

    def add_back_global_transform(self,seq,rots,trans):
        rots[:,[0,2]] = 0
        rot_mat = axis_angle_to_matrix(rots.squeeze())
        trans = trans[:,[1,0,2]]
        trans[:,1] *= -1

        offset = torch.mean(seq,dim=1,keepdim=True)
        trans = trans.unsqueeze(1)
        out_seq = seq - offset
        out_seq = torch.matmul(out_seq,rot_mat) + trans
        return out_seq

    def get_poisson_system(self,seq,faces,first_shape):
        poisson_matrices = poisson_system_matrices_from_mesh(first_shape.cpu().detach().numpy(),faces.cpu().detach().numpy())
        poisson_solver = poisson_matrices.create_poisson_solver().to('cuda')
        seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()

        return seq_jacobians,poisson_solver

    def get_poisson_system_from_shape(self,first_shape,faces):
        poisson_matrices = poisson_system_matrices_from_mesh(first_shape.cpu().detach().numpy(),faces.cpu().detach().numpy())
        poisson_solver = poisson_matrices.create_poisson_solver().to('cuda')

        return poisson_solver

    def get_centroids_normals(self,vertices,faces):
        centroids = torch.mean(vertices[faces],dim=1)
        normals = self.normals(vertices[faces,:])

        return centroids,normals
    
    def get_pointnet_features(self,normals,centroids):
        centroids_normals = torch.cat([normals,centroids],-1)
        centroids_normals_feat = self.pointnet_func(centroids_normals)

        return centroids_normals_feat
     
    def get_sequence_encoding_joints(self,pose,times):
        seq_encoding = self.joints_encoder.encoder(pose,times)
        return seq_encoding   

    def get_face_index_features(self):
        num_faces = 13776
        dims = 32
        indices_0to1 = torch.arange(0,1,step=(1.0/num_faces)).float().cuda()
        indices_0to1 = indices_0to1.unsqueeze(-1).expand(num_faces,dims)
        return indices_0to1

    def one_sequence(self,seq,faces,times,first_shape,poisson_solver,loss_args,train=True,test_shape=None,test_faces=None,pose=None,betas=None,clip_num=None,full_sequence_pose=None,full_sequence_times=None,d3f=None,no_gt=False):
        '''
        if self.poisson_solver is None:
            seq_jacobians,poisson_solver = self.get_poisson_system(seq,faces,first_shape)
            self.poisson_solver = poisson_solver
        else:
            poisson_solver = self.poisson_solver
            seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()
        '''
        if (train or not train) and not no_gt:
            seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()
        #print(seq[0])
        #mesh = trimesh.Trimesh(vertices=seq[0].cpu().detach().numpy(),faces=faces.cpu().detach().numpy(),process=False)
        #mesh.export("during_test.ply")
        #exit()
        first_j0 = poisson_solver.jacobians_from_vertices(first_shape.unsqueeze(0)).contiguous().squeeze()

        if train:
            centroids,normals = self.get_centroids_normals(first_shape,faces)
        else:
            if test_faces is not None:
                centroids,normals = self.get_centroids_normals(first_shape,test_faces)
            else:
                centroids,normals = self.get_centroids_normals(first_shape,faces)
        
        #cn_feat = d3f
        centroids_normals_cat = torch.cat([centroids,normals],-1) 
        cn_feat = self.pointnet_func(centroids_normals_cat)
        #cn_feat = self.get_face_index_features()
        #centroids_normals_cat = self.positional_encoding(centroids_normals_cat.unsqueeze(0).permute(1,0,2)).squeeze()
        pred_jacobians = self.posing_func(first_j0,cn_feat,d3f,pose,betas,times)
        #pred_jacobians = self.posing_func(first_j0,cn_feat,pose,betas)
        #if clip_num==1:
        #    print(first_j0)
        #    exit()
        if train:
            batch_xip1 = poisson_solver.solve_poisson(pred_jacobians) #.squeeze()
        else:
            batch_xip1 = poisson_solver.solve_poisson(pred_jacobians) #.squeeze()

        if train:
            deformed_shape,_,_ = shape_normalization_transforms_pytorch(batch_xip1)
        else:
            #deformed_shape,_,_ = shape_normalization_transforms_pytorch(batch_xip1)
            deformed_shape,_,_ = shape_normalization_transforms_pytorch(batch_xip1)
            #deformed_shape = batch_xip1

        if train:
            target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            gt_full_jacobians = seq_jacobians #.double()

            seq_loss_shape = loss_args.mse(deformed_shape,target_shape)
            v2v_errors,_ = compute_errors(deformed_shape,target_shape,faces)
            jac_loss = loss_args.mse(pred_jacobians,gt_full_jacobians)
            #print(torch.mean(v2v_errors))
            return seq_loss_shape,jac_loss,deformed_shape,v2v_errors,pred_jacobians,cn_feat
        else:
            #seq,_,_ = shape_normalization_transforms_pytorch(seq)
            #deformed_shape,_,_ = shape_normalization_transforms_pytorch(deformed_shape)
            if not no_gt:
                v2v_errors,_ = compute_errors(deformed_shape,seq,faces)
                gt_full_jacobians = seq_jacobians.view(seq_jacobians.size()[0],seq_jacobians.size()[1],-1)
                pred_jacobians = pred_jacobians.view(seq_jacobians.size()[0],seq_jacobians.size()[1],-1)
                l2j = torch.mean(torch.norm(pred_jacobians-gt_full_jacobians,dim=-1))
                mean_angle = 0.0
                for i in range(deformed_shape.size()[0]):
                    _pred = deformed_shape[i]
                    _gt = seq[i]
                    _,pred_n = self.get_centroids_normals(_pred,faces)
                    _,gt_n = self.get_centroids_normals(_gt,faces)

                    inner_product = (pred_n * gt_n).sum(dim=1)
                    #exit()
                    a_norm = torch.norm(pred_n,dim=1)
                    b_norm = torch.norm(gt_n,dim=1)
                    cos = inner_product #/ (2 * a_norm * b_norm)
                    angle = torch.rad2deg(torch.acos(cos))

                    if not torch.any(torch.isnan(angle)):
                        mean_angle += torch.mean(angle)
                mean_angle/=deformed_shape.size(0)

                return deformed_shape,v2v_errors,l2j,mean_angle
            else:
                return deformed_shape,pred_jacobians,cn_feat

    def amass_one_epoch(self,optimizer,epochs,loss_args,training_args,test_args,data_args,epoch):
        self.seq_dataset.shuffle()
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        mean_seq_loss = 0.0
        mean_jac_loss = 0.0
        mean_seq_v2v = 0.0
        num_sequences = 0.0
        logs_per_seq = 10
        prev_log = timer()
        logger = logging.getLogger("Running Epoch "+str(epoch))
        optimizer.zero_grad()
        batch_codes = []
        batch = 1
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0 

        seq_loss_dict = {}
        v2v_dict = {}
        for bix,data in enumerate(dataloader):
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,gender,_,name = data
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps)
            zero_betas = torch.zeros(betas.size()).float().cuda()
            root_zero = torch.zeros(root_orient.size()).float().cuda()

            if not training_args.root_zero:
                #primary_mean_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,zero_betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
                sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            else:
                #primary_mean_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,zero_betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
                sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            #primary_mean_sequence,wodmpl_sequence = align_feet(primary_mean_sequence,wodmpl_sequence)
            mean_sequence,_ = get_sequence(root_zero,pose_body,pose_hand,torch.zeros(trans.size()).float().cuda(),torch.zeros(betas.size()).float().cuda(),dmpls,2,data_args.max_frames,use_dmpl=False)
            first_shape = sequence[0]
            #primary_mean_first_shape = primary_mean_sequence[0]

            if not training_args.root_zero:
                sequence_pose = torch.cat([root_zero.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
            else:
                sequence_pose = torch.cat([root_zero.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)

            mean_seq_seq_loss = 0.0
            mean_seq_jac_loss = 0.0
            mean_seq_seq_v2v = 0.0
            _,clip_times,seq_times,split_indices = get_atomic_clips(mean_sequence,self.frame_length,fps)
            clip_poses,_,_ = split_by_indices(sequence_pose,seq_times,split_indices)
            clips,_,_ = split_by_indices(sequence,seq_times,split_indices)
            #wodmpl_clips,_,_ = split_by_indices(wodmpl_sequence,seq_times,split_indices)
            indices = list(range(len(clips)))
            clip_times = [clip_times[i] for i in indices]
            clip_poses = [clip_poses[i] for i in indices]
            clips = [clips[i] for i in indices]
            #primary_mean_clips = [primary_mean_clips[i] for i in indices]
            log_every = ceil(len(clips)*1.0/logs_per_seq)
            clip_ix = 0

            start_frame = 0
            feature_path = os.path.join(data_args.feature_dir,name[0]+".npy")
            feature_np = np.load(feature_path)
            face_features = torch.from_numpy(feature_np).float().cuda() #wks
            current_beta_poisson_solver = self.get_poisson_system_from_shape(first_shape,faces)
            for i,(clip,times,pose) in enumerate(zip(clips,clip_times,clip_poses)):
                shape_seq = clip
                #primmean_seq_loss,primmean_jac_loss,_ = self.one_sequence(primary_mean_shape_seq,faces,times,primary_mean_first_shape,loss_args,pose=pose,betas=zero_betas[0,0,0,:],full_sequence_pose=sequence_pose,full_sequence_times=seq_times)
                wodmpl_seq_loss,wodmpl_jac_loss,_,v2v_errors,_,_ = self.one_sequence(shape_seq,faces,times,first_shape,current_beta_poisson_solver,loss_args,d3f=face_features,pose=pose,betas=betas[0,0,0,:],full_sequence_pose=sequence_pose,full_sequence_times=seq_times)

                batch_seq_loss += (wodmpl_seq_loss) #primmean_seq_loss)
                batch_jac_loss += (wodmpl_jac_loss) #primmean_jac_loss)
                mean_seq_seq_loss += wodmpl_seq_loss.detach().item()
                mean_seq_jac_loss += wodmpl_jac_loss.detach().item()
                mean_seq_seq_v2v += v2v_errors.mean().cpu().detach().item()
                previous_times = times
                now = timer()

                if i%batch==0:
                    batch_seq_loss,batch_jac_loss = self.batch_backprop(optimizer,batch_seq_loss,batch_jac_loss)
                    batch_codes = []

                #if now - prev_log >= 60:
                if clip_ix == 0:
                    message = colored("{0:s} Sequence {1:3d} of {2:3d}".format(name[0],bix,len(dataloader)),"magenta")
                    logger.info(message)
                    #message = colored("Estimated {0:3.3f} Actual {1:3.3f} Speed {2:3.3f}".format(vid_l[0].item(),vid_l[1].item(),speed.item()),"red")
                    #logger.info(message)
                if  clip_ix % log_every == 0:
                    message = colored("Seq Loss: {0:2.6f} Jac loss: {1:2.6f} Clip: {2:4d} of {3:4d}"
                    .format(mean_seq_seq_loss/(clip_ix+1),mean_seq_jac_loss/(clip_ix+1),clip_ix,len(clips)),'cyan')
                    logger.info(message)

                    prev_log = now
                clip_ix +=1
                start_frame += len(clip)

            mean_seq_loss += (mean_seq_seq_loss/len(clips))
            mean_jac_loss += (mean_seq_jac_loss/len(clips))
            mean_seq_v2v += (mean_seq_seq_v2v/len(clips))
            seq_loss_dict[name[0]] = mean_seq_seq_loss/len(clips)
            v2v_dict[name[0]] = mean_seq_seq_v2v/len(clips)
            num_sequences += 1.0

            if len(batch_codes) > 0:
                if len(batch_codes) > 1:
                    batch_seq_loss,batch_jac_loss = self.batch_backprop(optimizer,batch_seq_loss,batch_jac_loss)
                    batch_codes = []
                else:
                    optimizer.zero_grad() ; self.pointnet_func.zero_grad() ; self.posing_func.zero_grad()

        for k,v in seq_loss_dict.items():
            v2v = v2v_dict[k]
            message = colored("{0:s} : {1:2.6f} {2:2.6f}".format(k,v,v2v),"yellow")
            logger.info(message)
        mean_seq_loss /= num_sequences            
        mean_jac_loss /= num_sequences
        mean_seq_v2v /= num_sequences

        return mean_seq_loss,mean_jac_loss,mean_seq_v2v

    def loop_epochs(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        scheduler = CosineAnnealingLR(optimizer,
                                    T_max = 300, # Maximum number of iterations.
                                    eta_min = 1e-4) # Minimum learning rate.

        #self.posing_func.load_state_dict(torch.load(training_args.weight_path+'_posing'))
        #self.pointnet_func.load_state_dict(torch.load(training_args.weight_path+'_features'))
        best_seq_loss = 1e9
        best_jac_loss = 1e9
        best_v2v = 1e9
        for i in range(training_args.epochs):
            logger = logging.getLogger("Finished Epoch "+str(i))
            #with torch.no_grad():
            #ep_seq_loss,ep_jac_loss,ep_v2v = self.amass_one_epoch_over_multi_betas(optimizer,epochs,loss_args,training_args,test_args,data_args,i)
            #with torch.no_grad():
            #    ep_seq_loss,ep_jac_loss,ep_v2v = self.amass_one_epoch_over_multi_betas_eval(optimizer,epochs,loss_args,training_args,test_args,data_args,i)
            ep_seq_loss,ep_jac_loss,ep_v2v = self.amass_one_epoch(optimizer,epochs,loss_args,training_args,test_args,data_args,i)
            #scheduler.step()
            if ep_seq_loss < best_seq_loss:
                best_seq_loss = ep_seq_loss
                best_epoch = i
                torch.save(self.posing_func.state_dict(),training_args.weight_path+'_posing')
                torch.save(self.pointnet_func.state_dict(),training_args.weight_path+'_features')

            if ep_jac_loss < best_jac_loss:
                best_jac_loss = ep_jac_loss

            if ep_v2v < best_v2v:
                best_v2v = ep_v2v

            message = colored("Best Ep: {0:3d} Best Seq Loss: {1:2.6f} Best Jac loss: {2:2.6f} Best v2v: {3:2.6f}"
            .format(best_epoch,best_seq_loss,best_jac_loss,best_v2v),'green')
            logger.info(message)
            #if best_seq_loss < 5e-4:
            #    break

    def def_trans_multiple(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        self.posing_func.load_state_dict(torch.load(training_args.weight_path+'_posing'))
        self.pointnet_func.load_state_dict(torch.load(training_args.weight_path+'_features'))
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)
        test_dataloader = DataLoader(self.test_seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        transfer_color = np.array([64,224,208],dtype=float)/255.0
        source_color = np.array([95,158,160],dtype=float)/255.0
        gt_color = np.array([175,225,175],dtype=float)/255.0
        root_dir = test_args.result_dir

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        all_v2v_errors = [] 
        all_v2p_errors = []
        all_l2j_errors = []
        all_l2n_errors = []
        name_and_v2p_errors = []
        name_and_v2v_errors = []
        name_and_l2j_errors = []
        name_and_l2n_errors = []
        with torch.no_grad():
            logger = logging.getLogger("Eval ")
            source_seq_ix = 0
            for bix,data in enumerate(dataloader):
                #if source_seq_ix==1:break
                root_orient,pose_body,pose_hand,trans,betas,_,dmpls,fps,gender,_,source_name = data
                test_seq_ix = 0
                for tbix,test_data in enumerate(test_dataloader):
                    _root_orient,_,_,_,test_betas,_,_,fps,test_gender,_,test_name = test_data
                    given_source_name = source_name[0].split("_")[0]
                    given_test_name = test_name[0].split("_")[0]
                    #if given_source_name == given_test_name:
                    #    continue
                    if not (given_source_name in test_args.src_names and given_test_name in test_args.tgt_names):
                        continue
                    test_betas = test_betas[:,:,0,:].unsqueeze(2)
                    test_betas = test_betas.repeat(1,1,trans.size()[1],1)
                    zero_betas = torch.zeros(test_betas.size()).float().cuda()
                    root_zero = torch.zeros(root_orient.size()).float().cuda()
                    root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps)
                    source_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)

                    sequence_pose = torch.cat([root_zero.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                    test_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=True)
                    #source_sequence,test_sequence = align_feet(source_sequence,test_sequence)
                    mean_sequence,_ = get_sequence(root_zero,pose_body,pose_hand,torch.zeros(trans.size()).float().cuda(),torch.zeros(betas.size()).float().cuda(),dmpls,2,data_args.max_frames,use_dmpl=False)
                    rest_shape = test_sequence[0]
                    _,source_clip_times,sequence_times,split_indices = get_atomic_clips(mean_sequence,self.frame_length,fps)
                    test_clips,_,_ = split_by_indices(test_sequence,sequence_times,split_indices)
                    source_clips,_,_ = split_by_indices(source_sequence,sequence_times,split_indices)
                    clip_poses,_,_ = split_by_indices(sequence_pose,sequence_times,split_indices)

                    start_frame = 0
                    clip_ix = 0
                    frame_ix = 0
                    message = colored("Source {0:3d} of {1:3d}, Test {2:3d} of {3:3d}".format(source_seq_ix,len(dataloader),test_seq_ix,len(test_dataloader)),'green')
                    logger.info(message)
                    pred_sequence = []
                    seq_v2p_error = 0.0
                    seq_v2v_error = 0.0
                    name = given_source_name + "_" + given_test_name
                    logger.info(name)

                    seq_total_errors = []
                    pred_seq = []
                    acc_seq_v2v_errors = []
                    gt_sequence = []

                    feature_path = os.path.join(data_args.feature_dir,test_name[0]+".npy")
                    feature_np = np.load(feature_path)
                    face_features = torch.from_numpy(feature_np).float().cuda() #wks
                    sequence_poisson_solver = self.get_poisson_system_from_shape(rest_shape,faces)

                    avg_l2j = 0.0 ; avg_l2n = 0.0
                    for i,(clip,test_clip,times,source_pose) in enumerate(zip(source_clips,test_clips,source_clip_times,clip_poses)):
                        trans_sequence,v2v_errors,l2js,l2ns  = self.one_sequence(clip,faces,times,rest_shape,sequence_poisson_solver,loss_args,d3f=face_features,train=False,pose=source_pose,betas=test_betas[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times)
                        #v2v_errors,v2p_errors = compute_errors(trans_sequence,test_clip,faces)
                        acc_seq_v2v_errors.append(v2v_errors)
                        seq_total_errors.append(v2v_errors)
                        pred_seq.append(trans_sequence)
                        gt_sequence.append(test_clip)
                        all_v2v_errors.append(torch.mean(v2v_errors).item())
                        #all_v2p_errors.append(torch.mean(v2p_errors).item())
                        frame_ix += len(clip)
                        #seq_v2p_error += torch.mean(v2p_errors).item()
                        seq_v2v_error += torch.mean(v2v_errors).item()
                        avg_l2j += l2js.item()
                        avg_l2n += l2ns.item()

                        #seq_name = str(start_frame) + "_" + str(start_frame+len(clip))
                        out_dir = os.path.join(root_dir,name,test_args.method)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        gt_out_dir = os.path.join(root_dir,name,"gt")
                        if not os.path.exists(gt_out_dir):
                            os.makedirs(gt_out_dir)
                        start_frame += len(clip) 
                        clip_ix += 1
                        message = colored("Rendered {0:3d} of {1:3d}".format(clip_ix,len(source_clips)),'blue')
                        logger.info(message)


                    seq_total_errors = torch.cat(seq_total_errors,0)
                    acc_seq_v2v_errors = torch.cat(acc_seq_v2v_errors,0)

                    start_frame = 0
                    if True: #test_args.sig:
                        for i,trans_sequence in enumerate(pred_seq):
                            #display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt",soft_errors=soft_v2v[start_frame:start_frame+len(trans_sequence)]) #v2v_errors=per_vertex_acc_errors[start_frame:start_frame+len(trans_sequence)])
                            #display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt",v2v_errors=acc_seq_v2v_errors[start_frame:start_frame+len(trans_sequence)],r90=True,angle=-90)
                            #display_utils.write_amass_sequence(gt_out_dir,"",gt_sequence[i],faces,start_frame,transfer_color,"tgt")
                            #display_utils.write_amass_sequence(out_dir,"",sk_clip,sk_faces,start_frame,source_color,"src")
                            start_frame+=len(trans_sequence)
                    
                    test_seq_ix += 1
                    seq_v2p_error /= len(source_clips)
                    seq_v2v_error /= len(source_clips)
                    avg_l2j /= len(source_clips)
                    avg_l2n /= len(source_clips)
                    all_l2j_errors.append(avg_l2j)
                    all_l2n_errors.append(avg_l2n)
                    name_and_v2p_errors.append([name,seq_v2p_error])
                    name_and_v2v_errors.append([name,seq_v2v_error])
                    name_and_l2j_errors.append([name,avg_l2j])
                    name_and_l2n_errors.append([name,avg_l2n])

                source_seq_ix += 1

            name_and_v2p_errors = sorted(name_and_v2p_errors,key=lambda x:x[1]) 
            name_and_v2v_errors = sorted(name_and_v2v_errors,key=lambda x:x[1]) 
            name_and_l2j_errors = sorted(name_and_l2j_errors,key=lambda x:x[1]) 
            name_and_l2n_errors = sorted(name_and_l2n_errors,key=lambda x:x[1]) 
            message = colored("v2v: {0:3.9f}".format(np.mean(all_v2v_errors)),'red')
            logger.info(message)
            message = colored("v2p: {0:3.9f}".format(np.mean(all_v2p_errors)),'red')
            logger.info(message)
            message = colored("l2j: {0:3.9f}".format(np.mean(all_l2j_errors)),'red')
            logger.info(message)
            message = colored("l2n: {0:3.9f}".format(np.mean(all_l2n_errors)),'red')
            logger.info(message)
            for item in name_and_v2p_errors:
                message = colored("Transfer: {0:s} v2p: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)
            for item in name_and_v2v_errors:
                message = colored("Transfer: {0:s} v2v: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)
            for item in name_and_l2j_errors:
                message = colored("Transfer: {0:s} l2j: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)
            for item in name_and_l2n_errors:
                message = colored("Transfer: {0:s} l2n: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)


    def _def_trans_multiple(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        self.posing_func.load_state_dict(torch.load(training_args.weight_path+'_posing'))
        self.pointnet_func.load_state_dict(torch.load(training_args.weight_path+'_features'))
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)
        test_dataloader = DataLoader(self.test_seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        transfer_color = np.array([64,224,208],dtype=float)/255.0
        source_color = np.array([95,158,160],dtype=float)/255.0
        gt_color = np.array([175,225,175],dtype=float)/255.0
        root_dir = test_args.result_dir

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        all_v2v_errors = [] 
        all_v2p_errors = []
        all_l2j_errors = []
        all_l2n_errors = []
        name_and_v2p_errors = []
        name_and_v2v_errors = []
        name_and_l2j_errors = []
        name_and_l2n_errors = []
        with torch.no_grad():
            logger = logging.getLogger("Eval ")
            source_seq_ix = 0
            for bix,data in enumerate(dataloader):
                #if source_seq_ix==1:break
                root_orient,pose_body,pose_hand,trans,betas,_,dmpls,fps,gender,_,source_name = data
                test_seq_ix = 0
                for tbix,test_data in enumerate(test_dataloader):
                    _root_orient,_,_,_,test_betas,_,_,fps,test_gender,_,test_name = test_data
                    given_source_name = source_name[0].split("_")[0]
                    given_test_name = test_name[0].split("_")[0]
                    #if given_source_name == given_test_name:
                    #    continue
                    if not (given_source_name in test_args.src_names and given_test_name in test_args.tgt_names):
                        continue
                    test_betas = test_betas[:,:,0,:].unsqueeze(2)
                    test_betas = test_betas.repeat(1,1,trans.size()[1],1)
                    zero_betas = torch.zeros(test_betas.size()).float().cuda()
                    root_zero = torch.zeros(root_orient.size()).float().cuda()
                    root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps)
                    source_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)

                    sequence_pose = torch.cat([root_zero.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                    test_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=True)
                    #source_sequence,test_sequence = align_feet(source_sequence,test_sequence)
                    rest_shape = test_sequence[0]
                    source_clips,source_clip_times,sequence_times,split_indices = get_atomic_clips(test_sequence,self.frame_length,fps)
                    test_clips,_,_ = split_by_indices(test_sequence,sequence_times,split_indices)
                    clip_poses,_,_ = split_by_indices(sequence_pose,sequence_times,split_indices)

                    start_frame = 0
                    clip_ix = 0
                    frame_ix = 0
                    message = colored("Source {0:3d} of {1:3d}, Test {2:3d} of {3:3d}".format(source_seq_ix,len(dataloader),test_seq_ix,len(test_dataloader)),'green')
                    logger.info(message)
                    pred_sequence = []
                    seq_v2p_error = 0.0
                    seq_v2v_error = 0.0
                    name = given_source_name + "_" + given_test_name
                    logger.info(name)

                    seq_total_errors = []
                    pred_seq = []
                    acc_seq_v2v_errors = []
                    gt_sequence = []

                    feature_path = os.path.join(data_args.feature_dir,test_name[0]+".npy")
                    feature_np = np.load(feature_path)
                    face_features = torch.from_numpy(feature_np).float().cuda() #wks
                    sequence_poisson_solver = self.get_poisson_system_from_shape(rest_shape,faces)

                    avg_l2j = 0.0; avg_l2n = 0.0
                    for i,(clip,test_clip,times,source_pose) in enumerate(zip(source_clips,test_clips,source_clip_times,clip_poses)):
                        trans_sequence,v2v_errors,l2js,l2ns = self.one_sequence(clip,faces,times,rest_shape,sequence_poisson_solver,loss_args,d3f=face_features,train=False,pose=source_pose,betas=test_betas[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times)
                        #v2v_errors,v2p_errors = compute_errors(trans_sequence,test_clip,faces)
                        acc_seq_v2v_errors.append(v2v_errors)
                        seq_total_errors.append(v2v_errors)
                        pred_seq.append(trans_sequence)
                        gt_sequence.append(test_clip)
                        all_v2v_errors.append(torch.mean(v2v_errors).item())
                        #all_v2p_errors.append(torch.mean(v2p_errors).item())
                        frame_ix += len(clip)
                        #seq_v2p_error += torch.mean(v2p_errors).item()
                        seq_v2v_error += torch.mean(v2v_errors).item()
                        avg_l2j += l2js.item()
                        avg_l2n += l2ns.item()
                        #seq_name = str(start_frame) + "_" + str(start_frame+len(clip))
                        out_dir = os.path.join(root_dir,name,test_args.method)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        gt_out_dir = os.path.join(root_dir,name,"gt")
                        if not os.path.exists(gt_out_dir):
                            os.makedirs(gt_out_dir)
                        start_frame += len(clip) 
                        clip_ix += 1
                        message = colored("Rendered {0:3d} of {1:3d}".format(clip_ix,len(source_clips)),'blue')
                        logger.info(message)


                    seq_total_errors = torch.cat(seq_total_errors,0)
                    acc_seq_v2v_errors = torch.cat(acc_seq_v2v_errors,0)

                    start_frame = 0
                    if True: #test_args.sig:
                        for i,trans_sequence in enumerate(pred_seq):
                            #display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt",soft_errors=soft_v2v[start_frame:start_frame+len(trans_sequence)]) #v2v_errors=per_vertex_acc_errors[start_frame:start_frame+len(trans_sequence)])
                            #display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt")
                            #display_utils.write_amass_sequence(gt_out_dir,"",gt_sequence[i],faces,start_frame,transfer_color,"tgt")
                            #display_utils.write_amass_sequence(out_dir,"",sk_clip,sk_faces,start_frame,source_color,"src")
                            start_frame+=len(trans_sequence)
                    
                    test_seq_ix += 1
                    seq_v2p_error /= len(source_clips)
                    seq_v2v_error /= len(source_clips)
                    avg_l2j /= len(source_clips)
                    avg_l2n /= len(source_clips)
                    all_l2j_errors.append(avg_l2j)
                    all_l2n_errors.append(avg_l2n)
                    name_and_v2p_errors.append([name,seq_v2p_error])
                    name_and_v2v_errors.append([name,seq_v2v_error])
                    name_and_l2j_errors.append([name,avg_l2j])
                    name_and_l2n_errors.append([name,avg_l2n])
                source_seq_ix += 1

            name_and_v2p_errors = sorted(name_and_v2p_errors,key=lambda x:x[1]) 
            name_and_v2v_errors = sorted(name_and_v2v_errors,key=lambda x:x[1]) 
            name_and_l2j_errors = sorted(name_and_l2j_errors,key=lambda x:x[1]) 
            name_and_l2n_errors = sorted(name_and_l2n_errors,key=lambda x:x[1]) 

            message = colored("v2v: {0:3.9f}".format(np.mean(all_v2v_errors)),'red')
            logger.info(message)
            message = colored("v2p: {0:3.9f}".format(np.mean(all_v2p_errors)),'red')
            logger.info(message)
            message = colored("l2j: {0:3.9f}".format(np.mean(all_l2j_errors)),'red')
            logger.info(message)
            message = colored("l2n: {0:3.9f}".format(np.mean(all_l2n_errors)),'red')
            logger.info(message)

            for item in name_and_v2p_errors:
                message = colored("Transfer: {0:s} v2p: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)
            for item in name_and_v2v_errors:
                message = colored("Transfer: {0:s} v2v: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)
            for item in name_and_l2j_errors:
                message = colored("Transfer: {0:s} l2j: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)
            for item in name_and_l2n_errors:
                message = colored("Transfer: {0:s} l2n: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)

    def def_trans_nonhuman(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        self.posing_func.load_state_dict(torch.load(training_args.weight_path+'_posing'))
        self.pointnet_func.load_state_dict(torch.load(training_args.weight_path+'_features'))
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)
        test_dataloader = DataLoader(self.test_seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        transfer_color = np.array([64,224,208],dtype=float)/255.0
        source_color = np.array([95,158,160],dtype=float)/255.0
        gt_color = np.array([175,225,175],dtype=float)/255.0
        root_dir = test_args.result_dir

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        nonhuman_mesh = trimesh.load(test_args.nonhuman_mesh,process=False)
        nonhuman_vertices = torch.from_numpy(np.array(nonhuman_mesh.vertices)).float().cuda()
        nonhuman_faces = torch.from_numpy(np.array(nonhuman_mesh.faces)).long().cuda()
        with torch.no_grad():
            logger = logging.getLogger("Eval ")
            source_seq_ix = 0
            for bix,data in enumerate(dataloader):
                #if source_seq_ix==1:break
                root_orient,pose_body,pose_hand,trans,betas,_,dmpls,fps,gender,_,source_name = data
                test_seq_ix = 0
                for tbix,test_data in enumerate(test_dataloader):
                    _root_orient,_,_,_,test_betas,_,_,fps,test_gender,_,test_name = test_data
                    given_source_name = source_name[0].split("_")[0]
                    given_test_name = test_name[0].split("_")[0]
                    #if given_source_name == given_test_name:
                    #    continue
                    if not (given_source_name in test_args.src_names and given_test_name in test_args.tgt_names):
                        continue
                    test_betas = test_betas[:,:,0,:].unsqueeze(2)
                    test_betas = test_betas.repeat(1,1,trans.size()[1],1)
                    root_zero = torch.zeros(root_orient.size()).float().cuda()
                    root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps)
                    source_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)

                    if not training_args.root_zero:
                        sequence_pose = torch.cat([root_zero.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                    else:
                        sequence_pose = torch.cat([root_zero.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)

                    if not training_args.root_zero:
                        test_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=True)
                    else:
                        test_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=True)

                    mean_sequence,_ = get_sequence(root_zero,pose_body,pose_hand,torch.zeros(trans.size()).float().cuda(),torch.zeros(betas.size()).float().cuda(),dmpls,2,data_args.max_frames,use_dmpl=False)
                    #source_sequence,test_sequence_wodmpl,test_sequence = align_feet(source_sequence,test_sequence_wodmpl,test_sequence)
                    rest_shape = nonhuman_vertices
                    _,source_clip_times,sequence_times,split_indices = get_atomic_clips(mean_sequence,self.frame_length,fps)
                    test_clips,_,_ = split_by_indices(test_sequence,sequence_times,split_indices)
                    source_clips,_,_ = split_by_indices(source_sequence,sequence_times,split_indices)
                    clip_poses,_,_ = split_by_indices(sequence_pose,sequence_times,split_indices)
                    clip_translations,_,_ = split_by_indices(trans.squeeze(),sequence_times,split_indices)
                    clip_rotations,_,_ = split_by_indices(root_orient.squeeze(),sequence_times,split_indices)

                    #_mesh = trimesh.Trimesh(vertices=rest_shape.cpu().detach().numpy(),faces=nonhuman_faces.cpu().detach().numpy(),process=False)
                    #_mesh.export('nonhuman.ply')
                    #_mesh = trimesh.Trimesh(vertices=test_sequence[0].cpu().detach().numpy(),faces=faces.cpu().detach().numpy(),process=False)
                    #_mesh.export('human.ply')
                    #exit()

                    start_frame = 0
                    clip_ix = 0
                    frame_ix = 0
                    message = colored("Source {0:3d} of {1:3d}, Test {2:3d} of {3:3d}".format(source_seq_ix,len(dataloader),test_seq_ix,len(test_dataloader)),'green')
                    logger.info(message)
                    nonhuman_name = test_args.nonhuman_mesh.split("/")[-1].split(".")[0]
                    name = given_source_name + "_" + test_args.nonhuman_mesh.split("/")[-1].split(".")[0]
                    logger.info(name)
                    pred_seq = []
                    prev_pred_jacobians = None
                    prev_pred_last_def = None
                    feature_path = os.path.join(test_args.nonhuman_features)
                    feature_np = np.load(feature_path)
                    face_features = torch.from_numpy(feature_np).float().cuda() #wks
                    #point_features = torch.from_numpy(feature_np).float().cuda()
                    #face_features = self.point_to_face_features(point_features,nonhuman_faces)
                    sequence_poisson_solver = self.get_poisson_system_from_shape(rest_shape,nonhuman_faces)
                    for i,(clip,test_clip,times,source_pose,g_rotation,g_trans) in enumerate(zip(source_clips,test_clips,source_clip_times,clip_poses,clip_rotations,clip_translations)):
                        trans_sequence,primary_jacobians,_ = self.one_sequence(clip,nonhuman_faces,times,rest_shape,sequence_poisson_solver,loss_args,d3f=face_features,pose=source_pose,betas=test_betas[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times,train=False,no_gt=True)
                        if True: #training_args.root_zero:
                            trans_sequence = self.add_back_global_transform(trans_sequence,g_rotation,g_trans)
                            trans_sequence = self.add_alignment_transform(trans_sequence,g_rotation,g_trans)
                        #trans_sequence,prev_pred_jacobians,prev_pred_last_def = self.one_sequence(clip,primary_jacobians,nonhuman_faces,times,rest_shape,sequence_poisson_solver,loss_args,train=False,pose=source_pose,betas=test_betas[0,0,0,:],full_sequence_pose=sequence_pose,full_sequence_times=sequence_times,clip_num=i,split_indices=split_indices,prev_pred_jacobians=prev_pred_jacobians,prev_pred_last_def=prev_pred_last_def,no_gt=True)
                        pred_seq.append(trans_sequence)
                        frame_ix += len(clip)

                        #seq_name = str(start_frame) + "_" + str(start_frame+len(clip))
                        #out_dir = os.path.join(root_dir,name) #test_args.method)
                        out_dir = os.path.join(root_dir,name,test_args.method+"_"+nonhuman_name)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        gt_out_dir = os.path.join(root_dir,name) #"gt")
                        if not os.path.exists(gt_out_dir):
                            os.makedirs(gt_out_dir)
                        start_frame += len(clip) 
                        clip_ix += 1
                        message = colored("Rendered {0:3d} of {1:3d}".format(clip_ix,len(source_clips)),'blue')
                        logger.info(message)

                    start_frame = 0
                    if True: #test_args.sig:
                        for i,trans_sequence in enumerate(pred_seq):
                            #display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt",soft_errors=soft_v2v[start_frame:start_frame+len(trans_sequence)]) #v2v_errors=per_vertex_acc_errors[start_frame:start_frame+len(trans_sequence)])
                            display_utils.write_amass_sequence(out_dir,"",trans_sequence,nonhuman_faces,start_frame,transfer_color,"tgt",r90=False) #,r90=True,angle=90)
                            #display_utils.write_amass_sequence(gt_out_dir,"",trans_sequence,nonhuman_faces,start_frame,transfer_color,"tgt")
                            #display_utils.write_amass_sequence(out_dir,"",sk_clip,sk_faces,start_frame,source_color,"src")
                            start_frame+=len(trans_sequence)
                    
                    test_seq_ix += 1
                source_seq_ix += 1