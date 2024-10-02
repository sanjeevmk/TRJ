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
from jacobian_network import GetNormals
from torchdiffeq import odeint_adjoint as odeint
from first_order_vertex_ode import VertexODE 
from util_networks import CustomPositionalEncoding
from posing_pretrain import PosingTrainer as Primary
from torch_scatter import scatter

class PatchTrainer():
    def __init__(self,posing_func,pointnet_func,vertex_deformation_func,transformer_network,joints_encoder,seq_dataset,batch,frame_length,logdir,test_dataset=None):
        self.seq_dataset = seq_dataset
        self.test_seq_dataset = test_dataset
        self.num_seqs = len(seq_dataset)
        self.batch = batch
        self.logdir = logdir
        self.posing_func = posing_func
        self.pointnet_func = pointnet_func
        self.vertex_deformation_func = vertex_deformation_func 
        self.joints_encoder = joints_encoder
        self.ode_evaluator = VertexODE(self.vertex_deformation_func)
        self.attention = transformer_network
        self.frame_length = frame_length
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.poisson_solver = None
        self.test_poisson_solver = None
        self.normals = GetNormals()
        self.method = 'euler'
        self.positional_encoding = CustomPositionalEncoding(6)
        self.primary = Primary(posing_func,pointnet_func,seq_dataset,batch,frame_length,logdir)

    def batch_backprop(self,optimizer,batch_seq_loss):
        loss = batch_seq_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.attention.zero_grad()
        self.posing_func.zero_grad()
        self.vertex_deformation_func.zero_grad()
        batch_seq_loss = 0.0

        return batch_seq_loss

    def get_poisson_system(self,seq,faces,first_shape):
        poisson_matrices = poisson_system_matrices_from_mesh(first_shape.cpu().detach().numpy(),faces.cpu().detach().numpy())
        poisson_solver = poisson_matrices.create_poisson_solver().to('cuda')
        seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()

        return seq_jacobians,poisson_solver

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

    def get_ode_solution_first_order(self,initial_delta,v0,primary_attention,prev_pred_attention,target_primary,edges,vertex_features,times,betas):
        num_verts = v0.size()[0]
        num_edges = edges.size()[0]
        primary_attention = primary_attention.clone().detach()
        prev_pred_attention = prev_pred_attention.clone().detach()
        num_verts_tensor = torch.ones(1,1).int().cuda()*num_verts
        num_edges_tensor = torch.ones(1,1).int().cuda()*num_edges
        initial_tensor = torch.cat([num_verts_tensor,num_edges_tensor,initial_delta.view(1,-1),v0.view(1,-1),primary_attention.view(1,-1),prev_pred_attention.view(1,-1),target_primary.view(1,-1),edges.view(1,-1),vertex_features.view(1,-1),betas.view(1,-1)],1)
        initial_state = tuple([initial_tensor])
        solution = odeint(self.ode_evaluator,initial_state,times,method=self.method)[0].squeeze(1)

        start_ix = 2 ; end_ix = start_ix +(num_verts*3)
        vert_first_solution_fetch_indices = torch.from_numpy(np.array(range(start_ix,end_ix))).type(torch.int64).unsqueeze(0).cuda()
        fetch_indices = vert_first_solution_fetch_indices.repeat(len(times),1)
        vertices = torch.gather(solution,1,fetch_indices).view(-1,num_verts,3).contiguous()

        return vertices 

    def get_sequence_encoding_joints(self,pose,times):
        seq_encoding = self.joints_encoder.encoder(pose,times)
        return seq_encoding   

    def get_poisson_system_from_shape(self,first_shape,faces):
        poisson_matrices = poisson_system_matrices_from_mesh(first_shape.cpu().detach().numpy(),faces.cpu().detach().numpy())
        poisson_solver = poisson_matrices.create_poisson_solver().to('cuda')

        return poisson_solver

    def get_vertex_features_from_face(self,cn_feat,faces):
        cn_feat_expanded = cn_feat.unsqueeze(1).repeat(1,3,1)
        cn_feat_flat = cn_feat_expanded.view(-1,cn_feat.size()[1])
        faces_flat = faces.view(-1)
        features = scatter(cn_feat_flat,faces_flat,dim=0,reduce='mean')

        return features

    def blend_predicted_shapes_with_target(self,pred_shapes,target_shape,times):
        norm_times = times/times[-1]
        blend_lambda = (1-norm_times).unsqueeze(-1).unsqueeze(-1)
        blend_pred = blend_lambda*pred_shapes
        blend_target = (1-blend_lambda)*target_shape.unsqueeze(0)
        blend = blend_pred + blend_target

        return blend

    def one_sequence(self,seq,primary_shape,cn_feat,faces,times,first_shape,poisson_solver,loss_args,train=True,test_shape=None,test_faces=None,pose=None,betas=None,full_sequence_pose=None,full_sequence_times=None,clip_num=None,split_indices=None,prev_pred_shapes=None,prev_pred_last_def=None):
        '''
        if self.poisson_solver is None:
            seq_jacobians,poisson_solver = self.get_poisson_system(seq,faces,first_shape)
            self.poisson_solver = poisson_solver
        else:
            poisson_solver = self.poisson_solver
            seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()
        '''
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)

        seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()
        first_j0 = poisson_solver.jacobians_from_vertices(first_shape.unsqueeze(0)).contiguous().squeeze()
        first_v0 = first_shape

        if train:
            centroids,normals = self.get_centroids_normals(first_shape,faces)
        else:
            if test_faces is not None:
                centroids,normals = self.get_centroids_normals(first_shape,test_faces)
            else:
                centroids,normals = self.get_centroids_normals(first_shape,faces)
        #centroids_normals_cat = torch.cat([centroids,normals],-1)
        #cn_feat = self.pointnet_func(centroids_normals_cat)
        #vertex_features = self.vertex_pointnet_func(first_v0)
        vertex_features = self.get_vertex_features_from_face(cn_feat,faces)
        #centroids_normals_cat = self.positional_encoding(centroids_normals_cat.unsqueeze(0).permute(1,0,2)).squeeze()

        #with torch.no_grad():
            #primary_jacobians = self.posing_func(first_j0,centroids_normals_cat,pose,torch.zeros(betas.size()).float().cuda())
        #    primary_jacobians = self.posing_func(first_j0,cn_feat,pose,torch.zeros(betas.size()).float().cuda())
        #    primary_shape = poisson_solver.solve_poisson(primary_jacobians) #.squeeze()
        #if clip_num==0:
        #    print(centroids_normals_cat)
        #    exit()
        #primary_jacobians = self.posing_func(first_j0,cn_feat,pose_encoding,torch.zeros(betas.size()).float().cuda())
        primary_shape_delta = primary_shape[1:] - primary_shape[:-1]
        primary_shapes_for_attention = primary_shape_delta.permute(1,0,2)
        primary_shapes_attention = self.attention(primary_shapes_for_attention,full_sequence_times[split_indices[clip_num]:split_indices[clip_num+1]-1])

        if clip_num==0:
            previous_shapes_attention = first_shape
            first_def = torch.zeros(first_shape.size()).float().cuda()
            #first_def = torch.eye(3).unsqueeze(0).repeat(first_j0.size()[0],1,1).float().cuda()
        else:
            prev_times = full_sequence_times[split_indices[clip_num-1]:split_indices[clip_num]]
            prev_shapes_delta = prev_pred_shapes[1:] - prev_pred_shapes[:-1]
            prev_shapes_for_attention = prev_shapes_delta.permute(1,0,2)
            previous_shapes_attention = self.attention(prev_shapes_for_attention,full_sequence_times[split_indices[clip_num-1]:split_indices[clip_num]-1])
            #first_def = torch.zeros(first_shape.size()).float().cuda()
            first_def = prev_pred_last_def.clone().detach()

        #deformation_jacobians = self.get_ode_solution_first_order(first_def,first_j0,primary_attention,previous_attention,centroids_normals_cat,times,betas)
        vertex_deformations = self.get_ode_solution_first_order(first_def,first_v0,primary_shapes_attention,previous_shapes_attention,primary_shape[-1],edges,vertex_features,times,betas)
        pred_shapes = primary_shape + vertex_deformations
        #pred_shapes = self.blend_predicted_shapes_with_target(pred_shapes,primary_shape[-1],times)
        last_def = vertex_deformations[-1]
        if train:
            deformed_shape,_,_ = shape_normalization_transforms_pytorch(pred_shapes)
        else:
            deformed_shape,_,_ = shape_normalization_transforms_pytorch(pred_shapes)
            #deformed_shape = pred_shapes

        if train:
            target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            v2v,_ = compute_errors(deformed_shape,target_shape,faces)

            seq_loss_shape = loss_args.mse(deformed_shape,target_shape)

            return seq_loss_shape,deformed_shape,last_def,v2v
        else:
            target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            v2v,_ = compute_errors(deformed_shape,target_shape,faces)
            return deformed_shape,last_def,v2v

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
            body_zero = torch.zeros(pose_body.size()).float().cuda()
            hand_zero = torch.zeros(pose_hand.size()).float().cuda()
            sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            first_shape = sequence[0]
            sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
            mean_seq_seq_loss = 0.0
            mean_seq_seq_v2v = 0.0
            clips,clip_times,seq_times,split_indices = get_atomic_clips(sequence,self.frame_length,fps)
            clip_poses,_,_ = split_by_indices(sequence_pose,seq_times,split_indices)
            indices = list(range(len(clips)))
            clip_times = [clip_times[i] for i in indices]
            clip_poses = [clip_poses[i] for i in indices]
            clips = [clips[i] for i in indices]
            log_every = ceil(len(clips)*1.0/logs_per_seq)
            clip_ix = 0

            start_frame = 0
            previous_pred_last_def = None
            previous_pred_shapes = None
            sequence_poisson_solver = self.get_poisson_system_from_shape(first_shape,faces)
            for i,(clip,times,pose) in enumerate(zip(clips,clip_times,clip_poses)):
                shape_seq = clip
                with torch.no_grad():
                    _,_,primary_shape,_,_,cn_feat = self.primary.one_sequence(shape_seq,faces,times,first_shape,sequence_poisson_solver,loss_args,pose=pose,betas=betas[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=seq_times)
                seq_loss,previous_pred_shapes,previous_pred_last_def,clip_v2v = self.one_sequence(shape_seq,primary_shape,cn_feat,faces,times,first_shape,sequence_poisson_solver,loss_args,pose=pose,betas=betas[0,0,0,:],full_sequence_pose=sequence_pose,full_sequence_times=seq_times,clip_num=i,split_indices=split_indices,prev_pred_shapes=previous_pred_shapes,prev_pred_last_def=previous_pred_last_def)


                batch_seq_loss += (seq_loss)
                mean_seq_seq_loss += seq_loss.detach().item()
                mean_seq_seq_v2v += clip_v2v.mean().cpu().detach().item()
                previous_times = times
                now = timer()

                if i%batch==0:
                    batch_seq_loss = self.batch_backprop(optimizer,batch_seq_loss)
                    batch_codes = []

                #if now - prev_log >= 60:
                if clip_ix == 0:
                    message = colored("{0:s} Sequence {1:3d} of {2:3d}".format(name[0],bix,len(dataloader)),"magenta")
                    logger.info(message)
                    #message = colored("Estimated {0:3.3f} Actual {1:3.3f} Speed {2:3.3f}".format(vid_l[0].item(),vid_l[1].item(),speed.item()),"red")
                    #logger.info(message)
                if  clip_ix % log_every == 0:
                    message = colored("Seq Loss: {0:2.6f} v2v: {1:2.6f} Clip: {2:4d} of {3:4d}"
                    .format(mean_seq_seq_loss/(clip_ix+1),mean_seq_seq_v2v/(clip_ix+1),clip_ix,len(clips)),'cyan')
                    logger.info(message)

                    prev_log = now
                clip_ix +=1
                start_frame += len(clip)

            mean_seq_loss += (mean_seq_seq_loss/len(clips))
            mean_seq_v2v += (mean_seq_seq_v2v/len(clips))
            seq_loss_dict[name[0]] = mean_seq_seq_loss/len(clips)
            v2v_dict[name[0]] = mean_seq_seq_v2v/len(clips)
            num_sequences += 1.0

            if len(batch_codes) > 0:
                if len(batch_codes) > 1:
                    #batch_seq_loss = self.batch_backprop(optimizer,batch_seq_loss)
                    batch_codes = []
                else:
                    optimizer.zero_grad() ; self.vertex_deformation_func.zero_grad() ; self.posing_func.zero_grad() ; self.attention.zero_grad()

        for k,v in seq_loss_dict.items():
            v2v = v2v_dict[k]
            message = colored("{0:s} : {1:2.6f} {2:2.6f}".format(k,v,v2v),"yellow")
            logger.info(message)
        mean_seq_loss /= num_sequences            
        mean_seq_v2v /= num_sequences            

        return mean_seq_loss,mean_seq_v2v

    def loop_epochs(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        scheduler = CosineAnnealingLR(optimizer,
                                    T_max = 300, # Maximum number of iterations.
                                    eta_min = 1e-4) # Minimum learning rate.
        self.primary.preload(training_args.pose_weights,training_args.feature_weights)

        #self.attention.load_state_dict(torch.load(training_args.weight_path+'_attention'))
        #self.vertex_deformation_func.load_state_dict(torch.load(training_args.weight_path+'_vertexdef'))
        best_seq_loss = 1e9
        best_v2v = 1e9
        for i in range(1000):
            logger = logging.getLogger("Finished Epoch "+str(i))
            ep_seq_loss,ep_v2v = self.amass_one_epoch(optimizer,epochs,loss_args,training_args,test_args,data_args,i)
            #scheduler.step()
            if ep_seq_loss < best_seq_loss:
                best_seq_loss = ep_seq_loss
                best_epoch = i
                #torch.save(self.posing_func.state_dict(),training_args.weight_path+'_posing')
                torch.save(self.attention.state_dict(),training_args.weight_path+'_attention')
                torch.save(self.vertex_deformation_func.state_dict(),training_args.weight_path+'_vertexdef')
                #torch.save(self.pointnet_func.state_dict(),training_args.weight_path+'_ode_features')
            if ep_v2v < best_v2v:
                best_v2v = ep_v2v


            message = colored("Best Ep: {0:3d} Best Seq Loss: {1:2.6f} Best v2v: {2:2.6f}"
            .format(best_epoch,best_seq_loss,best_v2v),'green')
            logger.info(message)

    def def_trans_multiple(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        self.primary.preload(training_args.pose_weights,training_args.feature_weights)
        self.vertex_deformation_func.load_state_dict(torch.load(training_args.weight_path+'_vertexdef'))
        self.attention.load_state_dict(torch.load(training_args.weight_path+'_attention'))

        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)
        test_dataloader = DataLoader(self.test_seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        transfer_color = np.array([64,224,208],dtype=float)/255.0
        source_color = np.array([95,158,160],dtype=float)/255.0
        gt_color = np.array([175,225,175],dtype=float)/255.0
        root_dir = test_args.result_dir

        '''
        for bix,data in enumerate(dataloader):
            #if source_seq_ix==1:break
            root_orient,pose_body,pose_hand,trans,betas,_,dmpls,fps,gender,_,source_name = data
            print(source_name,pose_body.size())
        exit()
        '''
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
                    root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps)
                    source_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)

                    sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                    test_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=True)
                    source_sequence,test_sequence = align_feet(source_sequence,test_sequence)
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
                    prev_pred_shapes = None
                    prev_pred_last_def = None
                    sequence_poisson_solver = self.get_poisson_system_from_shape(rest_shape,faces)
                    for i,(clip,test_clip,times,source_pose) in enumerate(zip(source_clips,test_clips,source_clip_times,clip_poses)):
                        _,_,primary_shape,_,_,cn_feat = self.primary.one_sequence(clip,faces,times,rest_shape,sequence_poisson_solver,loss_args,pose=source_pose,betas=betas[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times)
                        trans_sequence,prev_pred_last_def,v2v_errors = self.one_sequence(clip,primary_shape,cn_feat,faces,times,rest_shape,sequence_poisson_solver,loss_args,train=False,pose=source_pose,betas=betas[0,0,0,:],full_sequence_pose=sequence_pose,full_sequence_times=sequence_times,clip_num=i,split_indices=split_indices,prev_pred_shapes=prev_pred_shapes,prev_pred_last_def=prev_pred_last_def)
                        prev_pred_shapes = trans_sequence
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