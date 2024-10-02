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
from animals_utils import get_sequence,get_clips,get_time_for_clip,get_time_for_jacs,get_atomic_clips,split_by_indices
from timeit import default_timer as timer
import logging
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from math import ceil
from torch.optim.lr_scheduler import CosineAnnealingLR
from misc import compute_errors,align_feet,compute_soft,soft_displacements
from jacobian_network import GetNormals
from torchdiffeq import odeint_adjoint as odeint
from first_order_triangle_ode_d4d import TriangleODE
from util_networks import CustomPositionalEncoding
from posing_d4d import D4DPosingTrainer as Primary
from MeshProcessor import WaveKernelSignature
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_euler_angles
import math
from util_networks import BoneAngles

class D4DOdeTrainer():
    def __init__(self,posing_func,pointnet_func,jacobian_func,transformer_network,joints_encoder,seq_dataset,batch,frame_length,logdir,test_dataset=None):
        self.seq_dataset = seq_dataset
        self.test_seq_dataset = test_dataset
        self.num_seqs = len(seq_dataset)
        self.batch = batch
        self.logdir = logdir
        self.posing_func = posing_func
        self.pointnet_func = pointnet_func
        self.jacobian_func = jacobian_func
        self.joints_encoder = joints_encoder
        self.ode_evaluator = TriangleODE(self.jacobian_func)
        self.attention = transformer_network
        self.frame_length = frame_length
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.poisson_solver = None
        self.test_poisson_solver = None
        self.normals = GetNormals()
        self.method = 'euler'
        self.positional_encoding = CustomPositionalEncoding(6)
        self.bone_angles = BoneAngles()
        self.primary = Primary(posing_func,pointnet_func,seq_dataset,batch,frame_length,logdir)

    def batch_backprop(self,optimizer,batch_seq_loss,batch_jac_loss):
        loss = batch_seq_loss + 0.1*batch_jac_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.attention.zero_grad()
        self.posing_func.zero_grad()
        self.jacobian_func.zero_grad()
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0

        return batch_seq_loss,batch_jac_loss

    def get_poisson_system(self,seq,faces,first_shape):
        poisson_matrices = poisson_system_matrices_from_mesh(first_shape.cpu().detach().numpy(),faces.cpu().detach().numpy())
        poisson_solver = poisson_matrices.create_poisson_solver().to('cuda')
        seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()

        return seq_jacobians,poisson_solver

    def get_poisson_solver(self,faces,first_shape):
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

    def get_ode_solution_first_order(self,initial_delta,j0,primary_attention,prev_pred_attention,target_j,times,betas):
        num_faces = j0.size()[0]
        primary_attention = primary_attention.clone().detach()
        prev_pred_attention = prev_pred_attention.clone().detach()
        num_faces_tensor = torch.ones(1,1).int().cuda()*num_faces
        initial_tensor = torch.cat([num_faces_tensor,initial_delta.view(1,-1),j0.view(1,-1),primary_attention.view(1,-1),prev_pred_attention.view(1,-1),target_j.view(1,-1),betas.view(1,-1)],1)
        initial_state = tuple([initial_tensor])
        solution = odeint(self.ode_evaluator,initial_state,times,method=self.method)[0].squeeze(1)

        start_ix = 1 ; end_ix = start_ix +(num_faces*3*3)
        jac_first_solution_fetch_indices = torch.from_numpy(np.array(range(start_ix,end_ix))).type(torch.int64).unsqueeze(0).cuda()
        fetch_indices = jac_first_solution_fetch_indices.repeat(len(times),1)
        jac_first_order = torch.gather(solution,1,fetch_indices).view(-1,num_faces,3,3).contiguous()

        return jac_first_order

    def get_sequence_encoding_joints(self,pose,times):
        seq_encoding = self.joints_encoder.encoder(pose,times)
        return seq_encoding   

    def one_sequence(self,seq,primary_jacobians,faces,times,first_shape,poisson_solver,loss_args,train=True,test_shape=None,test_faces=None,pose=None,betas=None,full_sequence_pose=None,full_sequence_times=None,clip_num=None,split_indices=None,prev_pred_jacobians=None,prev_pred_last_def=None,no_gt=False,center=False):
        '''
        if self.poisson_solver is None:
            seq_jacobians,poisson_solver = self.get_poisson_system(seq,faces,first_shape)
            self.poisson_solver = poisson_solver
        else:
            poisson_solver = self.poisson_solver
            seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()
        '''

        if train:
            seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()
        first_j0 = poisson_solver.jacobians_from_vertices(first_shape.unsqueeze(0)).contiguous().squeeze()

        #centroids_normals_cat = torch.cat([centroids,normals],-1)
        #cn_feat = self.pointnet_func(centroids_normals_cat)
        #centroids_normals_cat = self.positional_encoding(centroids_normals_cat.unsqueeze(0).permute(1,0,2)).squeeze()

        #with torch.no_grad():
            #primary_jacobians = self.posing_func(first_j0,centroids_normals_cat,pose,torch.zeros(betas.size()).float().cuda())
            #primary_jacobians = self.posing_func(first_j0,cn_feat,pose,torch.zeros(betas.size()).float().cuda())
        #primary_jacobians = self.posing_func(first_j0,cn_feat,pose_encoding,torch.zeros(betas.size()).float().cuda())
        #primary_jacobians_delta = primary_jacobians[1:] - primary_jacobians[:-1]
        primary_jacobians_for_attention = primary_jacobians.view(times.size()[0],first_j0.size()[0],9).permute(1,0,2)
        primary_attention = self.attention(primary_jacobians_for_attention,full_sequence_times[split_indices[clip_num]:split_indices[clip_num+1]])

        if clip_num==0:
            previous_attention = first_j0
            first_def = torch.zeros(first_j0.size()).float().cuda()
            #first_def = primary_jacobians[1] - primary_jacobians[0]
            #first_def = torch.eye(3).unsqueeze(0).repeat(first_j0.size()[0],1,1).float().cuda()
        else:
            prev_times = full_sequence_times[split_indices[clip_num-1]:split_indices[clip_num]]
            #prev_jacobians_delta = prev_pred_jacobians[1:]-prev_pred_jacobians[:-1]
            prev_jacobians_for_attention = prev_pred_jacobians.view(prev_times.size()[0],first_j0.size()[0],9).permute(1,0,2)
            previous_attention = self.attention(prev_jacobians_for_attention,full_sequence_times[split_indices[clip_num-1]:split_indices[clip_num]])
            #first_def = torch.zeros(first_j0.size()).float().cuda()
            #first_def = primary_jacobians[1] - primary_jacobians[0]
            first_def = prev_pred_last_def.clone().detach()

        #deformation_jacobians = self.get_ode_solution_first_order(first_def,first_j0,primary_attention,previous_attention,centroids_normals_cat,times,betas)
        #pose_enc = self.get_sequence_encoding_joints(pose,times)
        deformation_jacobians = self.get_ode_solution_first_order(first_def,first_j0,primary_attention,previous_attention,primary_jacobians[-1].contiguous(),times,betas)
        pred_jacobians = primary_jacobians +deformation_jacobians
        #pred_jacobians = torch.matmul(primary_jacobians,deformation_jacobians)
        last_def = deformation_jacobians[-1]
        batch_xip1 = poisson_solver.solve_poisson(pred_jacobians) #.squeeze()
        if train:
            deformed_shape,_,_ = shape_normalization_transforms_pytorch(batch_xip1)
        else:
            if not center:
                deformed_shape = batch_xip1
            else:
                deformed_shape,_,_ = shape_normalization_transforms_pytorch(batch_xip1)

        if train:
            target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            v2v,_ = compute_errors(target_shape,deformed_shape,faces)
            gt_full_jacobians = seq_jacobians #.double()

            seq_loss_shape = loss_args.mse(deformed_shape,target_shape)
            jac_loss = loss_args.mse(pred_jacobians,gt_full_jacobians)

            return seq_loss_shape,jac_loss,deformed_shape,pred_jacobians,last_def,v2v
        else:
            if not no_gt:
                v2v,_ = compute_errors(seq,deformed_shape,faces)
                return deformed_shape,pred_jacobians,last_def,v2v
            else:
                return deformed_shape,pred_jacobians,last_def

    def dump_first_frames(self,training_args,data_args,test_args):
        self.seq_dataset.shuffle()
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        for bix,data in enumerate(dataloader):
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,gender,_,name = data
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps)
            sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            _mesh = trimesh.Trimesh(vertices=sequence[0].cpu().detach().numpy(),faces=faces.cpu().detach().numpy(),process=False)
            out_path = os.path.join(test_args.dataset_meshdir,name[0]+".ply")
            if not os.path.exists(test_args.dataset_meshdir):
                os.makedirs(test_args.dataset_meshdir)
            _mesh.export(out_path)

    def compute_and_write_wks(self,training_args,data_args,test_args):
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        for bix,data in enumerate(dataloader):
            sequence,faces,sig,name = data
            sequence = convert_to_float_tensors(sequence)[0].squeeze()
            faces = convert_to_long_tensors(faces)[0].squeeze()
            vertices = sequence[0]
            faces = faces[0]
            vertices = vertices.squeeze().cpu().detach().numpy()
            faces=faces.squeeze().cpu().detach().numpy()
            w = WaveKernelSignature(vertices, faces, top_k_eig=50)
            w.compute()
            wk = w.wks
            faces_wks = np.zeros((faces.shape[0], wk.shape[1]))
            for i in range(3):
                faces_wks += wk[faces[:, i], :]
            faces_wks /= 3
            out_path = os.path.join(data_args.feature_dir,name[0]+".npy")
            if not os.path.exists(data_args.feature_dir):
                os.makedirs(data_args.feature_dir)
            np.save(out_path,faces_wks)

    def point_to_face_features(self,features,faces):
        features_faces = features[faces]
        features_faces = torch.mean(features_faces,dim=1,keepdim=False)

        return features_faces

    def amass_one_epoch(self,optimizer,epochs,loss_args,training_args,test_args,data_args,epoch):
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
            vertices,faces,bone_pairs,betas,name = data
            sequence,betas = convert_to_float_tensors(vertices,betas)
            sequence = sequence.squeeze()
            betas = betas.squeeze()[0].unsqueeze(0)
            faces = convert_to_long_tensors(faces)[0].squeeze()
            bone_pairs = convert_to_long_tensors(bone_pairs)[0].squeeze()

            first_shape = sequence[0]
            seq_over_faces = sequence[:,faces[0],:]
            poses = self.bone_angles(seq_over_faces,bone_pairs)

            mean_seq_seq_loss = 0.0
            mean_seq_jac_loss = 0.0
            clips,clip_times,seq_times,split_indices = get_atomic_clips(sequence,self.frame_length)
            #random.shuffle(indices)
            seq_times_clips,_,_ = split_by_indices(seq_times,seq_times,split_indices)
            clips_poses,_,_ = split_by_indices(poses,seq_times,split_indices)
            #wodmpl_clips = [wodmpl_clips[i] for i in indices]
            log_every = ceil(len(clips)*1.0/logs_per_seq)
            clip_ix = 0

            start_frame = 0
            previous_pred_jacobians = None
            previous_pred_last_def = None
            mean_sequence_v2v = 0
            feature_path = os.path.join(data_args.feature_dir,name[0]+".npy")
            feature_np = np.load(feature_path)
            face_features = torch.from_numpy(feature_np).float().cuda() #wks
            #point_features = torch.from_numpy(feature_np).float().cuda() #d3f
            #face_features = self.point_to_face_features(point_features,faces)
            sequence_poisson_solver = self.get_poisson_system_from_shape(first_shape,faces[0])
            for i,(clip,times,seq_times_clip,pose) in enumerate(zip(clips,clip_times,seq_times_clips,clips_poses)):
                shape_seq = clip
                #with torch.no_grad():
                _,_,_,_,primary_jacobians = self.primary.one_sequence(shape_seq,faces[0],seq_times_clip,first_shape,sequence_poisson_solver,loss_args,d3f=face_features,pose=pose,betas=betas,clip_num=i)
                #primary_jacobians = sequence_poisson_solver.jacobians_from_vertices(wodmpl_clip).contiguous()
                seq_loss,jac_loss,_,previous_pred_jacobians,previous_pred_last_def,clip_v2v = self.one_sequence(shape_seq,primary_jacobians,faces[0],times,first_shape,sequence_poisson_solver,loss_args,pose=pose,betas=betas,full_sequence_times=seq_times,clip_num=i,split_indices=split_indices,prev_pred_jacobians=previous_pred_jacobians,prev_pred_last_def=previous_pred_last_def)

                batch_seq_loss += (seq_loss)
                batch_jac_loss += (jac_loss)
                mean_seq_seq_loss += seq_loss.detach().item()
                mean_seq_jac_loss += jac_loss.detach().item()
                mean_sequence_v2v += clip_v2v.mean().cpu().detach().item()
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
            mean_seq_v2v += (mean_sequence_v2v/len(clips))
            seq_loss_dict[name[0]] = mean_seq_seq_loss/len(clips)
            v2v_dict[name[0]] = mean_sequence_v2v/len(clips)
            num_sequences += 1.0

            if len(batch_codes) > 0:
                if len(batch_codes) > 1:
                    batch_seq_loss,batch_jac_loss = self.batch_backprop(optimizer,batch_seq_loss,batch_jac_loss)
                    batch_codes = []
                else:
                    optimizer.zero_grad() ; self.jacobian_func.zero_grad() ; self.posing_func.zero_grad() ; self.attention.zero_grad()

        for k,v in seq_loss_dict.items():
            seq_v2v = v2v_dict[k]
            message = colored("{0:s} : {1:2.6f} {2:2.6f}".format(k,v,seq_v2v),"yellow")
            logger.info(message)
        mean_seq_loss /= num_sequences            
        mean_jac_loss /= num_sequences
        mean_seq_v2v /= num_sequences

        return mean_seq_loss,mean_jac_loss,mean_seq_v2v

    def add_alignment_transform(self,seq,rots,trans):
        offset = torch.mean(seq,dim=1,keepdim=True)
        y90 = np.array([0,math.radians(90),0])
        y90 = torch.from_numpy(y90).float().cuda()
        rot_mat = axis_angle_to_matrix(y90)
        z180 = np.array([0,0,math.radians(180)])
        z180 = torch.from_numpy(z180).float().cuda()
        rot_mat_z180 = axis_angle_to_matrix(z180)
        #rot_mat = torch.matmul(rot_mat,rot_mat_z180)
        out_seq = seq - offset
        out_seq = torch.matmul(out_seq,rot_mat)
        out_seq += offset
        return out_seq

    def add_back_global_transform(self,seq,rots,trans):
        offset = torch.mean(seq,dim=1,keepdim=True)
        rot_mat = axis_angle_to_matrix(rots)
        trans = trans.unsqueeze(1)
        out_seq = seq - offset
        out_seq = torch.matmul(out_seq,rot_mat) + trans[:,:,[1,0,2]]
        return out_seq

    def get_poisson_system_from_shape(self,first_shape,faces):
        poisson_matrices = poisson_system_matrices_from_mesh(first_shape.cpu().detach().numpy(),faces.cpu().detach().numpy())
        poisson_solver = poisson_matrices.create_poisson_solver().to('cuda')

        return poisson_solver

    def loop_epochs(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        scheduler = CosineAnnealingLR(optimizer,
                                    T_max = 300, # Maximum number of iterations.
                                    eta_min = 1e-4) # Minimum learning rate.

        #self.dump_first_frames(training_args,data_args,test_args)
        #self.compute_and_write_wks(training_args,data_args,test_args)
        #exit()
        #self.primary.preload(training_args.pose_weights,training_args.feature_weights,training_args.weight_path)
        self.attention.load_state_dict(torch.load(training_args.weight_path+'_attention'))
        self.jacobian_func.load_state_dict(torch.load(training_args.weight_path+'_jacobian'))
        self.pointnet_func.load_state_dict(torch.load(training_args.weight_path+'_ode_features'))
        self.posing_func.load_state_dict(torch.load(training_args.weight_path+'_posing'))
        self.primary.overwrite_networks(self.posing_func,self.pointnet_func)
        #self.joints_encoder.load_state_dict(torch.load(training_args.weight_path+'_joints'))
        best_seq_loss = 1e9
        best_jac_loss = 1e9
        best_v2v = 1e9
        for i in range(1000):
            logger = logging.getLogger("Finished Epoch "+str(i))
            ep_seq_loss,ep_jac_loss,ep_v2v = self.amass_one_epoch(optimizer,epochs,loss_args,training_args,test_args,data_args,i)
            #scheduler.step()
            if ep_seq_loss < best_seq_loss:
                best_seq_loss = ep_seq_loss
                best_epoch = i
                torch.save(self.posing_func.state_dict(),training_args.weight_path+'_posing')
                torch.save(self.attention.state_dict(),training_args.weight_path+'_attention')
                torch.save(self.jacobian_func.state_dict(),training_args.weight_path+'_jacobian')
                #torch.save(self.joints_encoder.state_dict(),training_args.weight_path+'_joints')
                torch.save(self.pointnet_func.state_dict(),training_args.weight_path+'_ode_features')

            if ep_jac_loss < best_jac_loss:
                best_jac_loss = ep_jac_loss
            if ep_v2v < best_v2v:
                best_v2v = ep_v2v

            message = colored("Best Ep: {0:3d} Best Seq Loss: {1:2.6f} Best Jac loss: {2:2.6f} Best v2v: {3:2.6f}"
            .format(best_epoch,best_seq_loss,best_jac_loss,best_v2v),'green')
            logger.info(message)

    def def_trans_multiple(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        self.primary.preload(training_args.pose_weights,training_args.feature_weights,training_args.weight_path)
        self.jacobian_func.load_state_dict(torch.load(training_args.weight_path+'_jacobian'))
        self.attention.load_state_dict(torch.load(training_args.weight_path+'_attention'))
        #self.joints_encoder.load_state_dict(torch.load(training_args.weight_path+'_joints'))
        self.posing_func.load_state_dict(torch.load(training_args.weight_path+'_posing'))
        self.pointnet_func.load_state_dict(torch.load(training_args.weight_path+'_ode_features'))
        self.primary.overwrite_networks(self.posing_func,self.pointnet_func)
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
        name_and_v2p_errors = []
        name_and_v2v_errors = []
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
                    if not training_args.root_zero:
                        source_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
                    else:
                        source_sequence,faces = get_sequence(root_zero,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)

                    if not training_args.root_zero:
                        sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                    else:
                        sequence_pose = torch.cat([root_zero.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)

                    wodmpl_test_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=False)
                    test_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=True)
                    #source_sequence,test_sequence_wodmpl,test_sequence = align_feet(source_sequence,test_sequence_wodmpl,test_sequence)
                    rest_shape = test_sequence[0]
                    source_clips,source_clip_times,sequence_times,split_indices = get_atomic_clips(test_sequence,self.frame_length,fps)
                    test_clips,_,_ = split_by_indices(test_sequence,sequence_times,split_indices)
                    wodmpl_test_clips,_,_ = split_by_indices(wodmpl_test_sequence,sequence_times,split_indices)
                    clip_poses,_,_ = split_by_indices(sequence_pose,sequence_times,split_indices)
                    clip_translations,_,_ = split_by_indices(trans.squeeze(),sequence_times,split_indices)
                    clip_rotations,_,_ = split_by_indices(root_orient.squeeze(),sequence_times,split_indices)

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
                    prev_pred_jacobians = None
                    prev_pred_last_def = None
                    feature_path = os.path.join(data_args.feature_dir,test_name[0]+".npy")
                    feature_np = np.load(feature_path)
                    face_features = torch.from_numpy(feature_np).float().cuda()
                    sequence_poisson_solver = self.get_poisson_system_from_shape(rest_shape,faces)
                    for i,(clip,wodmpl_test_clip,test_clip,times,source_pose,g_rotation,g_trans) in enumerate(zip(source_clips,wodmpl_test_clips,test_clips,source_clip_times,clip_poses,clip_rotations,clip_translations)):
                        _,_,_,_,primary_jacobians = self.primary.one_sequence(clip,faces,times,rest_shape,sequence_poisson_solver,loss_args,d3f=face_features,pose=source_pose,betas=test_betas[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times)
                        trans_sequence,prev_pred_jacobians,prev_pred_last_def,v2v_errors = self.one_sequence(clip,primary_jacobians,faces,times,rest_shape,sequence_poisson_solver,loss_args,train=False,pose=source_pose,betas=test_betas[0,0,0,:],full_sequence_pose=sequence_pose,full_sequence_times=sequence_times,clip_num=i,split_indices=split_indices,prev_pred_jacobians=prev_pred_jacobians,prev_pred_last_def=prev_pred_last_def)
                        if training_args.root_zero:
                            trans_sequence = self.add_back_global_transform(trans_sequence,g_rotation,g_trans)
                        #for i in range(len(trans_sequence)):
                        #    _v = trans_sequence[i].cpu().detach().numpy()
                        #    _f = faces.cpu().detach().numpy()
                        #    _mesh = trimesh.Trimesh(vertices=_v,faces=_f,process=False)
                        #    _mesh.export("debug_global/"+str(i)+".ply")
                        #exit()
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
                            display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt")
                            display_utils.write_amass_sequence(gt_out_dir,"",gt_sequence[i],faces,start_frame,transfer_color,"tgt")
                            #display_utils.write_amass_sequence(out_dir,"",sk_clip,sk_faces,start_frame,source_color,"src")
                            start_frame+=len(trans_sequence)
                    
                    test_seq_ix += 1
                    seq_v2p_error /= len(source_clips)
                    seq_v2v_error /= len(source_clips)
                    name_and_v2p_errors.append([name,seq_v2p_error])
                    name_and_v2v_errors.append([name,seq_v2v_error])
                source_seq_ix += 1

            name_and_v2p_errors = sorted(name_and_v2p_errors,key=lambda x:x[1]) 
            name_and_v2v_errors = sorted(name_and_v2v_errors,key=lambda x:x[1]) 
            message = colored("v2v: {0:3.9f}".format(np.mean(all_v2v_errors)),'red')
            logger.info(message)
            message = colored("v2p: {0:3.9f}".format(np.mean(all_v2p_errors)),'red')
            logger.info(message)
            for item in name_and_v2p_errors:
                message = colored("Transfer: {0:s} v2p: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)
            for item in name_and_v2v_errors:
                message = colored("Transfer: {0:s} v2v: {1:3.9f}".format(item[0],item[1]),'magenta')
                logger.info(message)

    def def_trans_nonhuman(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        #self.primary.preload(training_args.pose_weights,training_args.feature_weights,training_args.weight_path)
        self.jacobian_func.load_state_dict(torch.load(training_args.weight_path+'_jacobian'))
        self.attention.load_state_dict(torch.load(training_args.weight_path+'_attention'))
        #self.joints_encoder.load_state_dict(torch.load(training_args.weight_path+'_joints'))
        self.posing_func.load_state_dict(torch.load(training_args.weight_path+'_posing'))
        self.pointnet_func.load_state_dict(torch.load(training_args.weight_path+'_ode_features'))
        self.primary.overwrite_networks(self.posing_func,self.pointnet_func)
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
                vertices,faces,bone_pairs,betas,source_name = data
                source_sequence,betas = convert_to_float_tensors(vertices,betas)
                source_sequence = source_sequence.squeeze()
                betas = betas.squeeze()
                betas = betas[0].unsqueeze(0)
                faces = convert_to_long_tensors(faces)[0].squeeze()
                bone_pairs = convert_to_long_tensors(bone_pairs)[0].squeeze()
                seq_over_faces = source_sequence[:,faces[0],:]
                poses = self.bone_angles(seq_over_faces,bone_pairs)

                test_seq_ix = 0
                for tbix,test_data in enumerate(test_dataloader):
                    _,_,_,_,test_name = data
                    given_source_name = source_name[0]
                    given_test_name = test_name[0]
                    #if given_source_name == given_test_name:
                    #    continue
                    if not (given_source_name in test_args.src_names and given_test_name in test_args.tgt_names):
                        continue

                    clips,clip_times,seq_times,split_indices = get_atomic_clips(source_sequence,self.frame_length)
                    #random.shuffle(indices)
                    seq_times_clips,_,_ = split_by_indices(seq_times,seq_times,split_indices)
                    clips_poses,_,_ = split_by_indices(poses,seq_times,split_indices)

                    rest_shape = nonhuman_vertices

                    '''
                    _mesh = trimesh.Trimesh(vertices=rest_shape.cpu().detach().numpy(),faces=nonhuman_faces.cpu().detach().numpy(),process=False)
                    _mesh.export('nonhuman.ply')
                    _mesh = trimesh.Trimesh(vertices=source_sequence[0].cpu().detach().numpy(),faces=faces[0].cpu().detach().numpy(),process=False)
                    _mesh.export('human.ply')
                    exit()
                    '''

                    start_frame = 0
                    clip_ix = 0
                    frame_ix = 0
                    message = colored("Source {0:3d} of {1:3d}, Test {2:3d} of {3:3d}".format(source_seq_ix,len(dataloader),test_seq_ix,len(test_dataloader)),'green')
                    logger.info(message)
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
                    gt = []
                    sequence_poisson_solver = self.get_poisson_system_from_shape(rest_shape,nonhuman_faces)
                    origin_translation = None
                    for i,(clip,times,seq_times_clip,pose) in enumerate(zip(clips,clip_times,seq_times_clips,clips_poses)):
                        _,primary_jacobians = self.primary.one_sequence(clip,nonhuman_faces,times,rest_shape,sequence_poisson_solver,loss_args,d3f=face_features,pose=pose,betas=betas,clip_num=i,train=False,no_gt=True)
                        trans_sequence,prev_pred_jacobians,prev_pred_last_def = self.one_sequence(clip,primary_jacobians,nonhuman_faces,times,rest_shape,sequence_poisson_solver,loss_args,train=False,pose=pose,betas=betas,full_sequence_times=seq_times,clip_num=i,split_indices=split_indices,prev_pred_jacobians=prev_pred_jacobians,prev_pred_last_def=prev_pred_last_def,no_gt=True,center=False)

                        pred_seq.append(trans_sequence)
                        gt.append(clip)
                        frame_ix += len(clip)


                        #seq_name = str(start_frame) + "_" + str(start_frame+len(clip))
                        out_dir = os.path.join(root_dir,name) #test_args.method)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        gt_out_dir = os.path.join(root_dir,name) #"gt")
                        if not os.path.exists(gt_out_dir):
                            os.makedirs(gt_out_dir)
                        start_frame += len(clip) 
                        clip_ix += 1
                        message = colored("Rendered {0:3d} of {1:3d}".format(clip_ix,len(clips)),'blue')
                        logger.info(message)

                    '''
                    for i in range(100):
                        _v = gt[i].cpu().detach().numpy()
                        _pv = pred[i].cpu().detach().numpy()
                        _f = faces.cpu().detach().numpy()
                        _pf = nonhuman_faces.cpu().detach().numpy()
                        _m = trimesh.Trimesh(vertices=_v,faces=_f,process=False)
                        _m.export("debug/"+"src"+str(i)+".ply")
                        _m = trimesh.Trimesh(vertices=_pv,faces=_pf,process=False)
                        _m.export("debug/"+"tgt"+str(i)+".ply")
                    exit()
                    '''
                    start_frame = 0
                    if True: #test_args.sig:
                        for i,trans_sequence in enumerate(pred_seq):
                            #display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt",soft_errors=soft_v2v[start_frame:start_frame+len(trans_sequence)]) #v2v_errors=per_vertex_acc_errors[start_frame:start_frame+len(trans_sequence)])
                            display_utils.write_amass_sequence(out_dir,"",trans_sequence,nonhuman_faces,start_frame,transfer_color,"tgt",r90=False)
                            #display_utils.write_amass_sequence(gt_out_dir,"",trans_sequence,nonhuman_faces,start_frame,transfer_color,"tgt")
                            #display_utils.write_amass_sequence(out_dir,"",sk_clip,sk_faces,start_frame,source_color,"src")
                            start_frame+=len(trans_sequence)
                    
                    test_seq_ix += 1
                source_seq_ix += 1