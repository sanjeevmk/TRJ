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

class PosingTrainer():
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

    def amass_one_epoch_over_multi_betas_eval(self,optimizer,epochs,loss_args,training_args,test_args,data_args,epoch):
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        mean_all_seq_loss = 0.0
        mean_all_jac_loss = 0.0
        mean_all_v2v = 0.0
        num_sequences = 0.0
        logs_per_seq = 10
        prev_log = timer()
        logger = logging.getLogger("Running Epoch "+str(epoch))
        optimizer.zero_grad()
        batch_codes = []
        batch = 1
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0 

        all_betas = [] ; all_genders = []
        for bix,data in enumerate(dataloader):
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,gender,_,name = data
            if "50002" not in name[0]:
                continue
            all_betas.append(betas[0,0,0,:].unsqueeze(0).unsqueeze(0).unsqueeze(0))
            all_genders.append(gender)
        seq_loss_dict = {}
        v2v_dict = {}

        for bix,data in enumerate(dataloader):
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,gender,_,name = data
            mean_beta_seq_loss = 0.0
            mean_beta_jac_loss = 0.0
            mean_beta_v2v_error = 0.0
            beta_index = 0
            for current_beta,current_gender in zip(all_betas,all_genders):
                root_orient,pose_body,pose_hand,trans,current_beta,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,current_beta,test_betas,dmpls,fps)
                wodmpl_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,current_beta,dmpls,current_gender,data_args.max_frames,use_dmpl=False)
                wodmpl_first_shape = wodmpl_sequence[0]
                sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)

                wodmpl_clips,clip_times,seq_times,split_indices = get_atomic_clips(wodmpl_sequence,self.frame_length,fps)
                clip_poses,_,_ = split_by_indices(sequence_pose,seq_times,split_indices)
                indices = list(range(len(wodmpl_clips)))
                clip_times = [clip_times[i] for i in indices]
                clip_poses = [clip_poses[i] for i in indices]
                wodmpl_clips = [wodmpl_clips[i] for i in indices]
                log_every = ceil(len(wodmpl_clips)*1.0/logs_per_seq)
                clip_ix = 0

                mean_current_beta_seq_loss = 0.0
                mean_current_beta_jac_loss = 0.0
                mean_current_beta_v2v_error = 0.0
                current_beta_poisson_solver = self.get_poisson_system_from_shape(wodmpl_first_shape,faces)
                for i,(wodmpl_clip,times,pose) in enumerate(zip(wodmpl_clips,clip_times,clip_poses)):
                    wodmpl_shape_seq = wodmpl_clip
                    wodmpl_seq_loss,wodmpl_jac_loss,_,v2v_errors,_,_ = self.one_sequence(wodmpl_shape_seq,faces,times,wodmpl_first_shape,current_beta_poisson_solver,loss_args,pose=pose,betas=current_beta[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=seq_times)

                    batch_seq_loss += (wodmpl_seq_loss)
                    batch_jac_loss += (wodmpl_jac_loss)
                    mean_current_beta_seq_loss += wodmpl_seq_loss.detach().item()
                    mean_current_beta_jac_loss += wodmpl_jac_loss.detach().item()
                    mean_current_beta_v2v_error += v2v_errors.mean().detach().item()

                    if clip_ix == 0:
                        message = colored("{0:s} Sequence {1:3d} of {2:3d}, Beta {3:3d} of {4:3d}".format(name[0],bix,len(dataloader),beta_index,len(all_betas)),"magenta")
                        logger.info(message)
                    if  clip_ix % log_every == 0:
                        message = colored("Seq Loss: {0:2.6f} Jac loss: {1:2.6f} Clip: {2:4d} of {3:4d}"
                        .format(mean_current_beta_seq_loss/(clip_ix+1),mean_current_beta_jac_loss/(clip_ix+1),clip_ix,len(wodmpl_clips)),'cyan')
                        logger.info(message)

                    clip_ix +=1
                beta_index+=1
                mean_beta_seq_loss += (mean_current_beta_seq_loss/len(wodmpl_clips))
                mean_beta_jac_loss += (mean_current_beta_jac_loss/len(wodmpl_clips))
                mean_beta_v2v_error += (mean_current_beta_v2v_error/len(wodmpl_clips))
                now = timer()

            mean_beta_seq_loss /= len(all_betas)
            mean_beta_jac_loss /= len(all_betas)
            mean_beta_v2v_error /= len(all_betas)

            mean_all_seq_loss += mean_beta_seq_loss
            mean_all_jac_loss += mean_beta_jac_loss
            mean_all_v2v += mean_beta_v2v_error

            seq_loss_dict[name[0]] = mean_beta_seq_loss
            v2v_dict[name[0]] = mean_beta_v2v_error
            num_sequences += 1.0



        for k,v in seq_loss_dict.items():
            v2v_error = v2v_dict[k]
            message = colored("{0:s} : {1:2.6f} {2:2.6f}".format(k,v,v2v_error),"yellow")
            logger.info(message)
        mean_all_seq_loss /= num_sequences            
        mean_all_jac_loss /= num_sequences
        mean_all_v2v /= num_sequences

        return mean_all_seq_loss,mean_all_jac_loss,mean_all_v2v

    def preload(self,pose_weights,feature_weights,weight_path):
        #self.posing_func.load_state_dict(torch.load(pose_weights))
        #self.pointnet_func.load_state_dict(torch.load(feature_weights))
        self.posing_func.load_state_dict(torch.load(weight_path+'_posing'))
        self.pointnet_func.load_state_dict(torch.load(weight_path+'_ode_features'))

    def overwrite_networks(self,posing_func,pointnet_func):
        self.posing_func = posing_func
        self.pointnet_func = pointnet_func

    def batch_backprop(self,optimizer,batch_seq_loss,batch_jac_loss):
        loss = batch_seq_loss + batch_jac_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.pointnet_func.zero_grad()
        self.posing_func.zero_grad()
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0

        return batch_seq_loss,batch_jac_loss

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

    def one_sequence(self,seq,faces,times,first_shape,poisson_solver,loss_args,train=True,test_shape=None,test_faces=None,pose=None,betas=None,clip_num=None,full_sequence_pose=None,full_sequence_times=None,no_gt=False):
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

        if train:
            centroids,normals = self.get_centroids_normals(first_shape,faces)
        else:
            if test_faces is not None:
                centroids,normals = self.get_centroids_normals(first_shape,test_faces)
            else:
                centroids,normals = self.get_centroids_normals(first_shape,faces)
        centroids_normals_cat = torch.cat([centroids,normals],-1)
        #pose_encoding = self.get_sequence_encoding_joints(pose,times)
        #with torch.no_grad():
        cn_feat = self.pointnet_func(centroids_normals_cat)
        #centroids_normals_cat = self.positional_encoding(centroids_normals_cat.unsqueeze(0).permute(1,0,2)).squeeze()
        pred_jacobians = self.posing_func(first_j0,cn_feat,pose,betas,times)
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
            deformed_shape = batch_xip1

        if train:
            target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            gt_full_jacobians = seq_jacobians #.double()

            seq_loss_shape = loss_args.mse(deformed_shape,target_shape)
            v2v_errors,_ = compute_errors(deformed_shape,target_shape,faces)
            jac_loss = loss_args.mse(pred_jacobians,gt_full_jacobians)
            #print(torch.mean(v2v_errors))
            return seq_loss_shape,jac_loss,deformed_shape,v2v_errors,pred_jacobians,cn_feat
        else:
            #target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            if not no_gt:
                v2v_errors,_ = compute_errors(deformed_shape,seq,faces)
                return deformed_shape,v2v_errors
            else:
                return deformed_shape,pred_jacobians

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
            primary_mean_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,zero_betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            wodmpl_sequence,_ = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            primary_mean_sequence,wodmpl_sequence = align_feet(primary_mean_sequence,wodmpl_sequence)
            wodmpl_first_shape = wodmpl_sequence[0]
            primary_mean_first_shape = primary_mean_sequence[0]
            sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
            mean_seq_seq_loss = 0.0
            mean_seq_jac_loss = 0.0
            mean_seq_seq_v2v = 0.0
            primary_mean_clips,clip_times,seq_times,split_indices = get_atomic_clips(primary_mean_sequence,self.frame_length,fps)
            clip_poses,_,_ = split_by_indices(sequence_pose,seq_times,split_indices)
            wodmpl_clips,_,_ = split_by_indices(wodmpl_sequence,seq_times,split_indices)
            indices = list(range(len(wodmpl_clips)))
            clip_times = [clip_times[i] for i in indices]
            clip_poses = [clip_poses[i] for i in indices]
            wodmpl_clips = [wodmpl_clips[i] for i in indices]
            primary_mean_clips = [primary_mean_clips[i] for i in indices]
            log_every = ceil(len(wodmpl_clips)*1.0/logs_per_seq)
            clip_ix = 0

            start_frame = 0
            feature_path = os.path.join(data_args.feature_dir,name[0]+".npy")
            feature_np = np.load(feature_path)
            face_features = torch.from_numpy(feature_np).float().cuda() #wks
            current_beta_poisson_solver = self.get_poisson_system_from_shape(wodmpl_first_shape,faces)
            for i,(wodmpl_clip,primary_mean_clip,times,pose) in enumerate(zip(wodmpl_clips,primary_mean_clips,clip_times,clip_poses)):
                wodmpl_shape_seq = wodmpl_clip
                primary_mean_shape_seq = primary_mean_clip
                #primmean_seq_loss,primmean_jac_loss,_ = self.one_sequence(primary_mean_shape_seq,faces,times,primary_mean_first_shape,loss_args,pose=pose,betas=zero_betas[0,0,0,:],full_sequence_pose=sequence_pose,full_sequence_times=seq_times)
                wodmpl_seq_loss,wodmpl_jac_loss,_,v2v_errors,_,_ = self.one_sequence(wodmpl_shape_seq,faces,times,wodmpl_first_shape,current_beta_poisson_solver,loss_args,d3f=face_features,pose=pose,betas=zero_betas[0,0,0,:],full_sequence_pose=sequence_pose,full_sequence_times=seq_times)

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
                    .format(mean_seq_seq_loss/(clip_ix+1),mean_seq_jac_loss/(clip_ix+1),clip_ix,len(wodmpl_clips)),'cyan')
                    logger.info(message)

                    prev_log = now
                clip_ix +=1
                start_frame += len(wodmpl_clip)

            mean_seq_loss += (mean_seq_seq_loss/len(wodmpl_clips))
            mean_jac_loss += (mean_seq_jac_loss/len(wodmpl_clips))
            mean_seq_v2v += (mean_seq_seq_v2v/len(wodmpl_clips))

            seq_loss_dict[name[0]] = mean_seq_seq_loss/len(wodmpl_clips)
            v2v_dict[name[0]] = mean_seq_seq_v2v/len(wodmpl_clips)
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

    def amass_one_epoch_over_multi_betas(self,optimizer,epochs,loss_args,training_args,test_args,data_args,epoch):
        #self.seq_dataset.shuffle()
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        mean_all_seq_loss = 0.0
        mean_all_jac_loss = 0.0
        mean_all_v2v = 0.0
        num_sequences = 0.0
        logs_per_seq = 10
        prev_log = timer()
        logger = logging.getLogger("Running Epoch "+str(epoch))
        optimizer.zero_grad()
        batch_codes = []
        batch = 1
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0 

        all_betas = [] ; all_genders = []
        for bix,data in enumerate(dataloader):
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,gender,_,name = data
            #if "50002" not in name[0]:
            #    continue
            all_betas.append(betas[0,0,0,:].unsqueeze(0).unsqueeze(0).unsqueeze(0))
            all_genders.append(gender)
        seq_loss_dict = {}
        v2v_dict = {}

        for bix,data in enumerate(dataloader):
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,gender,_,name = data
            mean_beta_seq_loss = 0.0
            mean_beta_jac_loss = 0.0
            mean_beta_v2v_error = 0.0
            beta_index = 0
            for current_beta,current_gender in zip(all_betas,all_genders):
                root_orient,pose_body,pose_hand,trans,current_beta,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,current_beta,test_betas,dmpls,fps)
                wodmpl_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,current_beta,dmpls,current_gender,data_args.max_frames,use_dmpl=False)
                wodmpl_first_shape = wodmpl_sequence[0]
                sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)

                wodmpl_clips,clip_times,seq_times,split_indices = get_atomic_clips(wodmpl_sequence,self.frame_length,fps)
                clip_poses,_,_ = split_by_indices(sequence_pose,seq_times,split_indices)
                indices = list(range(len(wodmpl_clips)))
                clip_times = [clip_times[i] for i in indices]
                clip_poses = [clip_poses[i] for i in indices]
                wodmpl_clips = [wodmpl_clips[i] for i in indices]
                log_every = ceil(len(wodmpl_clips)*1.0/logs_per_seq)
                clip_ix = 0

                mean_current_beta_seq_loss = 0.0
                mean_current_beta_jac_loss = 0.0
                mean_current_beta_v2v_error = 0.0
                current_beta_poisson_solver = self.get_poisson_system_from_shape(wodmpl_first_shape,faces)
                #r_ix = random.choice(list(range(len(wodmpl_clips))))
                #for i,(wodmpl_clip,times,pose) in enumerate(zip(wodmpl_clips[r_ix:r_ix+1],clip_times[r_ix:r_ix+1],clip_poses[r_ix:r_ix+1])):
                for i,(wodmpl_clip,times,pose) in enumerate(zip(wodmpl_clips,clip_times,clip_poses)):
                    wodmpl_shape_seq = wodmpl_clip
                    wodmpl_seq_loss,wodmpl_jac_loss,_,v2v_errors,_,_ = self.one_sequence(wodmpl_shape_seq,faces,times,wodmpl_first_shape,current_beta_poisson_solver,loss_args,pose=pose,betas=current_beta[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=seq_times)

                    batch_seq_loss += (wodmpl_seq_loss)
                    batch_jac_loss += (wodmpl_jac_loss)
                    mean_current_beta_seq_loss += wodmpl_seq_loss.detach().item()
                    mean_current_beta_jac_loss += wodmpl_jac_loss.detach().item()
                    mean_current_beta_v2v_error += v2v_errors.mean().detach().item()

                    if i%batch==0:
                        batch_seq_loss,batch_jac_loss = self.batch_backprop(optimizer,batch_seq_loss,batch_jac_loss)

                    if clip_ix == 0:
                        message = colored("{0:s} Sequence {1:3d} of {2:3d}, Beta {3:3d} of {4:3d}".format(name[0],bix,len(dataloader),beta_index,len(all_betas)),"magenta")
                        logger.info(message)
                    if  clip_ix % log_every == 0:
                        message = colored("Seq Loss: {0:2.6f} Jac loss: {1:2.6f} Clip: {2:4d} of {3:4d}"
                        .format(mean_current_beta_seq_loss/(clip_ix+1),mean_current_beta_jac_loss/(clip_ix+1),clip_ix,len(wodmpl_clips)),'cyan')
                        logger.info(message)

                    clip_ix +=1
                beta_index+=1
                mean_beta_seq_loss += (mean_current_beta_seq_loss/len(wodmpl_clips))
                mean_beta_jac_loss += (mean_current_beta_jac_loss/len(wodmpl_clips))
                mean_beta_v2v_error += (mean_current_beta_v2v_error/len(wodmpl_clips))
                now = timer()

            mean_beta_seq_loss /= len(all_betas)
            mean_beta_jac_loss /= len(all_betas)
            mean_beta_v2v_error /= len(all_betas)

            mean_all_seq_loss += mean_beta_seq_loss
            mean_all_jac_loss += mean_beta_jac_loss
            mean_all_v2v += mean_beta_v2v_error

            seq_loss_dict[name[0]] = mean_beta_seq_loss
            v2v_dict[name[0]] = mean_beta_v2v_error
            num_sequences += 1.0



        for k,v in seq_loss_dict.items():
            v2v_error = v2v_dict[k]
            message = colored("{0:s} : {1:2.6f} {2:2.6f}".format(k,v,v2v_error),"yellow")
            logger.info(message)
        mean_all_seq_loss /= num_sequences            
        mean_all_jac_loss /= num_sequences
        mean_all_v2v /= num_sequences

        return mean_all_seq_loss,mean_all_jac_loss,mean_all_v2v

    def loop_epochs(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        scheduler = CosineAnnealingLR(optimizer,
                                    T_max = 300, # Maximum number of iterations.
                                    eta_min = 1e-4) # Minimum learning rate.

        #self.posing_func.load_state_dict(torch.load(training_args.weight_path+'_posing'))
        #self.pointnet_func.load_state_dict(torch.load(training_args.weight_path+'_features'))
        best_seq_loss = 1e9
        best_jac_loss = 1e9
        best_v2v = 1e9
        for i in range(1000):
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
                    source_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=False)

                    sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                    test_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=False)
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
                    sequence_poisson_solver = self.get_poisson_system_from_shape(rest_shape,faces)

                    for i,(clip,test_clip,times,source_pose) in enumerate(zip(source_clips,test_clips,source_clip_times,clip_poses)):
                        trans_sequence,v2v_errors = self.one_sequence(clip,faces,times,rest_shape,sequence_poisson_solver,loss_args,train=False,pose=source_pose,betas=test_betas[0,0,0,:],clip_num=i,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times)
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