from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn
import torch
import numpy as np
import trimesh
import os
import pickle
from torch.autograd.functional import jacobian
from torch.linalg import eigh
import torch.nn.functional as F
from termcolor import colored
import time
import display_utils
import igl
from datetime import datetime,timezone
#from pytorch3d.transforms import rotation_conversions,standardize_quaternion
from PoissonSystem import poisson_system_matrices_from_mesh
from misc import shape_normalization_transforms_pytorch,get_current_time,convert_to_float_tensors,convert_to_float_tensors_from_numpy,convert_to_long_tensors_from_numpy,convert_to_long_tensors
import csv
from torch.autograd import Variable
import random
random.seed(10)
from networks import GetNormals
from torch.utils.data import DataLoader
from pytorch3d.ops import knn_points
from triangle_ode import TriangleODE
from first_order_triangle_ode import TriangleODE as TriangleODEFirstOrder
from pytorch3d.loss import chamfer_distance
from gaussian_loss import NormalReg
from amass_utils import get_sequence,get_clips,get_time_for_clip,get_time_for_jacs,get_atomic_clips,get_pose_of_shape,split_by_indices,get_atomic_clips_by_joints
from timeit import default_timer as timer
import logging
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from math import ceil
from networks import JacobianIntegrator
from torch.optim.lr_scheduler import CosineAnnealingLR
from misc import compute_errors,align_feet,compute_soft,soft_displacements
from networks import CustomPositionalEncoding
from skeleton_sequence import sequence_to_skeleton_sequence

class Method():
    def __init__(self,tangent_func,seq_dataset,n_dims,batch,logdir,mesh_logdir,frame_length,transformer_network=None,pointnet_network=None,acceleration_network=None,test_dataset=None,joint_network=None,beta_network=None,projection_network=None,decoder_network=None):
        self.jacobian_first_order = tangent_func
        self.acceleration_func = acceleration_network 
        self.jacobian_second_order = JacobianIntegrator()
        self.seq_dataset = seq_dataset
        self.test_seq_dataset = test_dataset
        self.projection_network = projection_network
        self.num_seqs = len(seq_dataset)
        self.n_dims = n_dims
        self.batch = batch
        self.triangle_func = TriangleODE(self.jacobian_first_order,self.acceleration_func,self.jacobian_second_order,self.n_dims)
        self.triangle_func_first_order = TriangleODEFirstOrder(self.jacobian_first_order,self.acceleration_func,self.jacobian_second_order,self.n_dims)
        self.method = 'euler'
        self.zeros = torch.tensor(0).float().cuda()
        self.ones = torch.tensor(1).float().cuda()
        self.logdir = logdir
        self.mesh_logdir = mesh_logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.transformer_network = transformer_network
        self.pointnet_network = pointnet_network
        self.joint_network = joint_network
        self.regloss = NormalReg(regWeight=1.0)
        self.minus_one = torch.tensor(-1).unsqueeze(0).float().cuda()
        self.normals = GetNormals()
        self.frame_length = frame_length
        self.blending_length = 128
        self.beta_network = beta_network
        self.positional_encoding = CustomPositionalEncoding(16)
        self.poisson_solver = None
        self.test_poisson_solver = None
        self.decoder_network = decoder_network 

    def reload_if_needed(self,optimizer,acc_norm,first_norm,training_args):
        if acc_norm>=0.05 or first_norm>=0.05:
            self.pointnet_network.load_state_dict(torch.load(training_args.weight_path+'_pointnet'))
            self.jacobian_first_order.load_state_dict(torch.load(training_args.weight_path+'_tangent'))
            self.transformer_network.load_state_dict(torch.load(training_args.weight_path+'_transformer'))
            self.acceleration_func.load_state_dict(torch.load(training_args.weight_path+'_acceleration'))
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2.0

    def normalize_one_minus_one(self,tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)

        #tensor = 2*((tensor - min_val)/(max_val-min_val+1e-9)) - 1
        tensor = (tensor - min_val)/(max_val-min_val+1e-9)

        return tensor

    def batch_backprop_first_order(self,optimizer,batch_seq_loss,batch_jac_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss,training_args,epoch=0):
        loss = batch_seq_loss + batch_jac_loss + 0.0*batch_ae_loss + 0.0*batch_gaussian_loss + 0.0*batch_beta_loss
        #loss = batch_seq_loss + batch_seq_loss_first_order + 0.0*batch_jac_loss + 0.0*batch_jac_first_order_loss + 0.00*batch_jac_velocity_loss + 0.0*batch_ae_loss + 0.01*batch_gaussian_loss + 0.01*batch_beta_loss
        loss.backward()
        total_acc_norm = 0.0
        for p in self.acceleration_func.parameters():
            param_norm = p.grad.detach().data.norm(2).item()
            total_acc_norm += param_norm**2
        total_acc_norm = total_acc_norm**0.5
        total_first_norm = 0.0
        for p in self.jacobian_first_order.parameters():
            param_norm = p.grad.detach().data.norm(2).item()
            total_first_norm += param_norm**2
        total_first_norm = total_first_norm**0.5
        print(colored("Ep: {0:3d} Acc: {1:2.6f} First:{2:2.6f}".format(epoch,total_acc_norm,total_first_norm),"red"))
        optimizer.step()
        optimizer.zero_grad()
        self.pointnet_network.zero_grad()
        self.jacobian_first_order.zero_grad()
        self.transformer_network.zero_grad()
        self.acceleration_func.zero_grad()
        self.beta_network.zero_grad()
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0 ; batch_ae_loss = 0.0 ; batch_gaussian_loss = 0.0
        batch_beta_loss = 0.0 

        #if epoch>0:
        #    self.reload_if_needed(optimizer,total_acc_norm,total_first_norm,training_args)
        return batch_seq_loss,batch_jac_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss

    def batch_backprop(self,optimizer,batch_seq_loss,batch_seq_loss_first_order,batch_jac_loss,batch_jac_first_order_loss,batch_jac_velocity_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss,training_args,epoch=0):
        #loss = batch_seq_loss + 0.01*batch_seq_loss_first_order + 0.1*batch_jac_loss + 0.01*batch_jac_first_order_loss + batch_jac_velocity_loss + 0.0*batch_ae_loss + 0.0*batch_gaussian_loss + 0.01*batch_beta_loss
        loss = batch_seq_loss + 0.1*batch_jac_loss + batch_jac_velocity_loss
        #loss = batch_seq_loss + batch_seq_loss_first_order + 0.0*batch_jac_loss + 0.0*batch_jac_first_order_loss + 0.00*batch_jac_velocity_loss + 0.0*batch_ae_loss + 0.01*batch_gaussian_loss + 0.01*batch_beta_loss
        loss.backward()
        total_acc_norm = 0.0
        for p in self.acceleration_func.parameters():
            param_norm = p.grad.detach().data.norm(2).item()
            total_acc_norm += param_norm**2
        total_acc_norm = total_acc_norm**0.5
        total_first_norm = 0.0
        for p in self.jacobian_first_order.parameters():
            param_norm = p.grad.detach().data.norm(2).item()
            total_first_norm += param_norm**2
        total_first_norm = total_first_norm**0.5
        print(colored("Ep: {0:3d} Acc: {1:2.6f} First:{2:2.6f}".format(epoch,total_acc_norm,total_first_norm),"red"))
        optimizer.step()
        optimizer.zero_grad()
        self.pointnet_network.zero_grad()
        self.jacobian_first_order.zero_grad()
        self.transformer_network.zero_grad()
        self.acceleration_func.zero_grad()
        self.beta_network.zero_grad()
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0 ; batch_jac_first_order_loss = 0.0 ; batch_jac_velocity_loss = 0.0 ; batch_ae_loss = 0.0 ; batch_gaussian_loss = 0.0
        batch_beta_loss = 0.0 ; batch_seq_loss_first_order = 0.0

        #if epoch>0:
        #    self.reload_if_needed(optimizer,total_acc_norm,total_first_norm,training_args)
        return batch_seq_loss,batch_seq_loss_first_order,batch_jac_loss,batch_jac_first_order_loss,batch_jac_velocity_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss

    def blend_predicted_jacobians_with_target(self,pred_jacobians,target_jacobian,times):
        norm_times = times/times[-1]
        blend_lambda = (1-norm_times).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        blend_pred = blend_lambda*pred_jacobians
        blend_target = (1-blend_lambda)*target_jacobian
        blend = blend_pred + blend_target

        return blend

    def get_poisson_system(self,seq,faces):
        poisson_matrices = poisson_system_matrices_from_mesh(seq[0].cpu().detach().numpy(),faces.cpu().detach().numpy())
        poisson_solver = poisson_matrices.create_poisson_solver().to('cuda')
        seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()

        return seq_jacobians,poisson_solver

    def get_centroids_normals(self,vertices,faces):
        centroids = torch.mean(vertices[faces],dim=1)
        normals = self.normals(vertices[faces,:])

        return centroids,normals
    
    def get_pointnet_features(self,normals,centroids):
        centroids_normals = torch.cat([normals,centroids],-1)
        centroids_normals_feat = self.pointnet_network(centroids_normals)

        return centroids_normals_feat

    def get_initial_velocity(self,jacobians,times):
        j_t1 = jacobians[1:2,:,:,:]
        j_t0 = jacobians[0:1,:,:,:]
        norm_times = times/times[-1]
        #j_t1 = j_t0 + (j_t1-j_t0)*1e-9
        times_t1 = times[1:2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,jacobians.size()[1],3,3)
        times_t0 = times[0:1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,jacobians.size()[1],3,3)
        #times_t1 = times_t0 + (times_t1-times_t0)*1e-9
        jac_delta = j_t1 - j_t0
        time_delta = times_t1 - times_t0
        dj_dt = torch.div(jac_delta,time_delta)
        return dj_dt

    def get_gt_velocity(self,jacobians,times):
        j_t1 = jacobians[1:,:,:,:]
        j_t0 = jacobians[:-1,:,:,:]
        #norm_times = times/times[-1]
        times_t1 = times[1:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,jacobians.size()[1],3,3)
        times_t0 = times[:-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,jacobians.size()[1],3,3)
        jac_delta = j_t1 - j_t0
        time_delta = times_t1 - times_t0
        dj_dt = torch.div(jac_delta,time_delta)
        #dj_dt = jac_delta
        #dj_dt_t0 = torch.zeros(1,dj_dt.size()[1],3,3).float().cuda()
        #dj_dt = torch.cat([dj_dt_t0,dj_dt],0)
        return dj_dt

    def get_ode_solution_first_order(self,j0,target_j,centroids_normals,seq_encoding,times,betas,spf,centroids_normals_cat):
        num_faces = j0.size()[0]
        j0_second_order = j0.clone().detach()
        j1 = target_j.clone().detach()
        num_faces_tensor = torch.ones(1,1).int().cuda()*num_faces
        spf_tensor = torch.ones(1,1).float().cuda()*spf
        tmax = torch.ones(1,1).float().cuda()*times[-1].item()
        initial_tensor = torch.cat([num_faces_tensor,spf_tensor,tmax,j0.view(1,-1),centroids_normals.view(1,-1),j0.view(1,-1),j1.view(1,-1),betas.view(1,-1),centroids_normals_cat.view(1,-1),seq_encoding.view(1,-1),times.unsqueeze(0)],1)
        initial_state = tuple([initial_tensor])
        solution = odeint(self.triangle_func_first_order,initial_state,times,method=self.method,options={"reverse_time":False})[0].squeeze(1)

        start_ix = 3 ; end_ix = start_ix +(num_faces*3*3)
        jac_first_solution_fetch_indices = torch.from_numpy(np.array(range(start_ix,end_ix))).type(torch.int64).unsqueeze(0).cuda()
        fetch_indices = jac_first_solution_fetch_indices.repeat(len(times),1)
        jac_first_order = torch.gather(solution,1,fetch_indices).view(-1,num_faces,3,3).contiguous()

        return jac_first_order

    def get_ode_solution(self,j0,velocity_t0,target_j,centroids_normals,seq_encoding,times,betas,spf,centroids_normals_cat):
        num_faces = j0.size()[0]
        j0_second_order = j0.clone().detach()
        j1 = target_j.clone().detach()
        num_faces_tensor = torch.ones(1,1).int().cuda()*num_faces
        spf_tensor = torch.ones(1,1).float().cuda()*spf
        tmax = torch.ones(1,1).float().cuda()*times[-1].item()

        initial_tensor = torch.cat([num_faces_tensor,spf_tensor,tmax,j0.view(1,-1),velocity_t0.view(1,-1),j0_second_order.view(1,-1),centroids_normals.view(1,-1),j0.view(1,-1),velocity_t0.view(1,-1),j1.view(1,-1),betas.view(1,-1),centroids_normals_cat.view(1,-1),seq_encoding.view(1,-1),times.unsqueeze(0)],1)
        initial_state = tuple([initial_tensor])
        solution = odeint(self.triangle_func,initial_state,times,method=self.method,options={"reverse_time":False})[0].squeeze(1)

        start_ix = 3 ; end_ix = start_ix +(num_faces*3*3)
        jac_first_solution_fetch_indices = torch.from_numpy(np.array(range(start_ix,end_ix))).type(torch.int64).unsqueeze(0).cuda()
        fetch_indices = jac_first_solution_fetch_indices.repeat(len(times),1)
        jac_first_order = torch.gather(solution,1,fetch_indices).view(-1,num_faces,3,3).contiguous()

        start_ix = end_ix ; end_ix = start_ix + (num_faces*3*3)
        dj_dt_fetch_indices = torch.from_numpy(np.array(range(start_ix,end_ix))).type(torch.int64).unsqueeze(0).cuda()
        fetch_indices = dj_dt_fetch_indices.repeat(len(times),1)
        dj_dt = torch.gather(solution,1,fetch_indices).view(-1,num_faces,3,3).contiguous()

        start_ix = end_ix ; end_ix = start_ix + (num_faces*3*3)
        jac_second_order_fetch_indices = torch.from_numpy(np.array(range(start_ix,end_ix))).type(torch.int64).unsqueeze(0).cuda()
        fetch_indices = jac_second_order_fetch_indices.repeat(len(times),1)
        jac_second_order = torch.gather(solution,1,fetch_indices).view(-1,num_faces,3,3).contiguous()
        return jac_first_order,dj_dt,jac_second_order

    def get_sequence_encoding_joints(self,pose,times):
        seq_encoding = self.transformer_network.encoder(pose,times)
        return seq_encoding

    def get_beta(self,seq_jacobians,centroids_normals_cat,times):
        centroids_normals_for_transformer = centroids_normals_cat.unsqueeze(1).repeat(1,times.size()[0],1)
        seq_jacobians_for_transformer = seq_jacobians.permute(1,0,2,3).view(seq_jacobians.size()[1],seq_jacobians.size()[0],-1)
        random_jac_indices = np.array(list(range(len(times)))) #+1
        #random_jac_indices = np.array([-1])
        #random_jac_indices = np.array(list(range(len(times)-4,len(times)))) #+1
        betas = self.beta_network.encoder(seq_jacobians_for_transformer[:,random_jac_indices,:],centroids_normals_for_transformer,times[random_jac_indices])
        #betas = self.beta_network.encoder(seq_jacobians_for_transformer[:,[-1],:])
        return betas 

    def get_sequence_encoding(self,seq_jacobians_for_transformer,times):
        random_jac_indices = np.array(list(range(len(times)))) #+1
        #random_jac_indices = np.array([-1])
        #random_jac_indices = np.array(list(range(len(times)-4,len(times)))) #+1
        seq_encoding = self.transformer_network.encoder(seq_jacobians_for_transformer[:,random_jac_indices,:],times[random_jac_indices])
        return seq_encoding

    def get_sequence_decoding(self,code,j0,triangles_0):
        decoded_delta_jacobians = self.transformer_network.decoder(code,j0,triangles_0)

        return decoded_delta_jacobians

    def one_sequence_first_order(self,seq,faces,times,first_shape,loss_args,spf=None,gt_gauss_code=None,train=True,test_shape=None,last_shape=None,test_faces=None,poisson_solver=None,test_poisson_solver=None,pose=None,betas=None,encoding=None,full_sequence_pose=None,full_sequence_times=None,clip_num=0,previous_seq=None,previous_pred_seq=None,previous_times=None,split_indices=None):
        if self.poisson_solver is None:
            seq_jacobians,poisson_solver = self.get_poisson_system(seq,faces)
            self.poisson_solver = poisson_solver
            first_j0 = poisson_solver.jacobians_from_vertices(first_shape.unsqueeze(0)).contiguous().squeeze()
        else:
            poisson_solver = self.poisson_solver
            seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()
            first_j0 = poisson_solver.jacobians_from_vertices(first_shape.unsqueeze(0)).contiguous().squeeze()

        if not train:
            if test_faces is None:
                if self.test_poisson_solver is None:
                    test_shape_jacobians,test_poisson_solver = self.get_poisson_system(test_shape,faces)
                    self.test_poisson_solver = test_poisson_solver
                else:
                    test_poisson_solver = self.test_poisson_solver
                    test_shape_jacobians = test_poisson_solver.jacobians_from_vertices(test_shape).contiguous()
            else:
                if self.test_poisson_solver is None:
                    test_shape_jacobians,test_poisson_solver = self.get_poisson_system(test_shape,test_faces)
                    self.test_poisson_solver = test_poisson_solver
                else:
                    test_poisson_solver = self.test_poisson_solver
                    test_shape_jacobians = test_poisson_solver.jacobians_from_vertices(test_shape).contiguous()

            if last_shape is not None:
                target_jacobians = test_poisson_solver.jacobians_from_vertices(last_shape).contiguous()
            #test_shape_jacobians = test_poisson_solver.jacobians_from_vertices(test_shape).contiguous()

        if train:
            centroids,normals = self.get_centroids_normals(first_shape,faces)
        else:
            if test_faces is not None:
                centroids,normals = self.get_centroids_normals(first_shape,test_faces)
            else:
                centroids,normals = self.get_centroids_normals(first_shape,faces)
        centroids_normals_cat = torch.cat([centroids,normals],-1)
        centroids_normals = self.get_pointnet_features(normals,centroids)

        seq_jacobians_for_transformer = seq_jacobians.view(len(times),faces.size()[0],9).permute(1,0,2)
        seq_encoding = self.get_sequence_encoding_joints(full_sequence_pose,full_sequence_times)

        if train:
            j0 = seq_jacobians[0]
            target_j = seq_jacobians[-1].contiguous()
        else:
            j0 = test_shape_jacobians[0]
            target_j = target_jacobians

        if train:
            #pred_j = self.decoder_network(seq_encoding,first_j0,full_sequence_times[split_indices[clip_num+1]])
            jac_first_order = self.get_ode_solution_first_order(j0,target_j,centroids_normals,seq_encoding,times,betas,spf,centroids_normals_cat)
        else:
            jac_first_order = self.get_ode_solution_first_order(j0,target_j,centroids_normals,seq_encoding,times,betas,spf,centroids_normals_cat)

        pred_full_jacobians = jac_first_order
        if train:
            batch_xip1 = poisson_solver.solve_poisson(pred_full_jacobians) #.squeeze()
            pred_beta = self.get_beta(pred_full_jacobians,centroids_normals_cat,times)
        else:
            #blending version next 2 lines uncomment
            #pred_full_jacobians = jac_second_order
            #blended_jacobians = self.blend_predicted_jacobians_with_target(pred_full_jacobians,target_jacobians,times)
            #batch_xip1 = test_poisson_solver.solve_poisson(blended_jacobians) #.squeeze()
            batch_xip1 = test_poisson_solver.solve_poisson(pred_full_jacobians) #.squeeze()

        if train:
            deformed_shape,_,_ = shape_normalization_transforms_pytorch(batch_xip1)
        else:
            deformed_shape = batch_xip1
        if train:
            target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            gt_full_jacobians = seq_jacobians #.double()

            gt_ae_jacobians = seq_jacobians_for_transformer[:,[-1],:].view(1,faces.size()[0],3,3).squeeze()
            dj_dt_gt = self.get_gt_velocity(seq_jacobians,times)
            seq_loss_shape = loss_args.mse(deformed_shape,target_shape)
            jac_loss = loss_args.mse(pred_full_jacobians,gt_full_jacobians)
            gt_betas = betas.unsqueeze(0).unsqueeze(0)
            gt_betas = self.normalize_one_minus_one(gt_betas)
            beta_loss = loss_args.mse(pred_beta.squeeze(),gt_betas)
            autoencoder_loss = loss_args.mse(target_j,gt_ae_jacobians)

            dj_dt = []
            jac_velocity_loss = torch.zeros(1).float().cuda()
            my_euler_jacobians = [j0.unsqueeze(0)]
            my_euler_djdt = [] 
            with torch.no_grad():
                print(times)
                for tix in range(len(times)):
                    if tix==0:
                        curr = j0.view(j0.size()[0],9).clone().detach()
                    else:
                        curr = dj_dt[-1].view(j0.size()[0],9).clone().detach()
                    _dj_dt = self.jacobian_first_order(centroids_normals,seq_encoding,j0.view(j0.size()[0],9),betas,None,curr,None,None,times[tix],None)
                    if tix > 0:
                        _my_euler = my_euler_jacobians[-1][0] + (times[tix]-times[tix-1])*dj_dt[-1][0]
                        _my_dj_dt = torch.div(_my_euler - my_euler_jacobians[-1][0],times[tix]-times[tix-1])
                        my_euler_jacobians.append(_my_euler.unsqueeze(0))
                        my_euler_djdt.append(_my_dj_dt.unsqueeze(0))
                    dj_dt.append(_dj_dt.view(-1,3,3).unsqueeze(0))
                dj_dt = torch.cat(dj_dt,0)
                my_euler_djdt = torch.cat(my_euler_djdt,0)
                my_euler_jacobians = torch.cat(my_euler_jacobians,0)
                #print(my_euler_jacobians.size(),pred_full_jacobians.size())
                #exit()
                jac_velocity_loss = loss_args.mse(dj_dt[:-1],dj_dt_gt)
                #jac_velocity_loss = loss_args.mse(my_euler_jacobians,pred_full_jacobians)
                #jac_velocity_loss = loss_args.mse(my_euler_djdt,dj_dt_gt)
                #print(dj_dt[1][:10].view(-1,9))
                #print(dj_dt_gt[1][:10].view(-1,9))
            #print()
            #print()
            #jac_velocity_loss = loss_args.mse(dj_dt[:-1],dj_dt_gt)
            #jac_velocity_loss = l1loss(dj_dt[:-1],dj_dt_gt)
            #dj_dt_gt = torch.cat([velocity_t0,dj_dt_gt],0)
            #if clip_num == 0:
            #    v0 = torch.zeros(dj_dt_gt[0].size()).unsqueeze(0).float().cuda()
            #    dj_dt_gt = torch.cat([v0,dj_dt_gt])
            #else:
            #previous_gt_jacobians = poisson_solver.jacobians_from_vertices(previous_seq).contiguous()
            #previous_velocities = self.get_gt_velocity(previous_gt_jacobians,previous_times)
            #v0 = previous_velocities[-1].unsqueeze(0).contiguous()
            #jac_velocity_loss = l1loss(dj_dt[1:],dj_dt_gt)
            #gaussian_loss = loss_args.mse(seq_encoding,gt_gauss_code)

            return seq_loss_shape,jac_loss,jac_velocity_loss,beta_loss,autoencoder_loss,seq_encoding,deformed_shape
        else:
            return deformed_shape

    def one_sequence(self,seq,faces,times,first_shape,loss_args,spf=None,gt_gauss_code=None,train=True,test_shape=None,last_shape=None,test_faces=None,poisson_solver=None,test_poisson_solver=None,pose=None,betas=None,encoding=None,full_sequence_pose=None,full_sequence_times=None,clip_num=0,previous_seq=None,previous_pred_seq=None,previous_times=None,split_indices=None):
        if self.poisson_solver is None:
            seq_jacobians,poisson_solver = self.get_poisson_system(seq,faces)
            self.poisson_solver = poisson_solver
            first_j0 = poisson_solver.jacobians_from_vertices(first_shape.unsqueeze(0)).contiguous().squeeze()
        else:
            poisson_solver = self.poisson_solver
            seq_jacobians = poisson_solver.jacobians_from_vertices(seq).contiguous()
            first_j0 = poisson_solver.jacobians_from_vertices(first_shape.unsqueeze(0)).contiguous().squeeze()

        train_target_jacobians = poisson_solver.jacobians_from_vertices(seq[-1].unsqueeze(0)).contiguous()
        #seq_jacobians = self.blend_predicted_jacobians_with_target(seq_jacobians,target_jacobians,times)
        l1loss = nn.MSELoss()
        #betas = self.normalize_one_minus_one(betas)
        if not train:
            if test_faces is None:
                if self.test_poisson_solver is None:
                    test_shape_jacobians,test_poisson_solver = self.get_poisson_system(test_shape,faces)
                    self.test_poisson_solver = test_poisson_solver
                else:
                    test_poisson_solver = self.test_poisson_solver
                    test_shape_jacobians = test_poisson_solver.jacobians_from_vertices(test_shape).contiguous()
            else:
                if self.test_poisson_solver is None:
                    test_shape_jacobians,test_poisson_solver = self.get_poisson_system(test_shape,test_faces)
                    self.test_poisson_solver = test_poisson_solver
                else:
                    test_poisson_solver = self.test_poisson_solver
                    test_shape_jacobians = test_poisson_solver.jacobians_from_vertices(test_shape).contiguous()

            if last_shape is not None:
                target_jacobians = test_poisson_solver.jacobians_from_vertices(last_shape).contiguous()
            #test_shape_jacobians = test_poisson_solver.jacobians_from_vertices(test_shape).contiguous()
        #if train:
        #    centroids,normals = self.get_centroids_normals(seq[0,:,:],faces)
        #    triangles_v_0 = seq[0,:,:][faces] #.view(-1,9)
        #else:

        if train:
            centroids,normals = self.get_centroids_normals(first_shape,faces)
        else:
            if test_faces is not None:
                centroids,normals = self.get_centroids_normals(first_shape,test_faces)
            else:
                centroids,normals = self.get_centroids_normals(first_shape,faces)
        centroids_normals_cat = torch.cat([centroids,normals],-1)
        centroids_normals = self.get_pointnet_features(normals,centroids)

        seq_jacobians_for_transformer = seq_jacobians.view(len(times),faces.size()[0],9).permute(1,0,2)
        if encoding!="joints":
            seq_encoding = self.get_sequence_encoding(seq_jacobians_for_transformer,times)
        else:
            #seq_encoding = self.get_sequence_encoding_joints(pose,times)
            #seq_encoding = self.get_sequence_encoding_joints(pose,times)
            seq_encoding = self.get_sequence_encoding_joints(full_sequence_pose,full_sequence_times)

        if train:
            j0 = seq_jacobians[0]
            target_j = seq_jacobians[-1].contiguous()
        else:
            source_j0 = seq_jacobians[0]
            j0 = test_shape_jacobians[0]
            target_j = target_jacobians
        #velocity_t0 = self.get_initial_velocity(seq_jacobians,times)
        #velocity_t0 = torch.eye(3).float().cuda()
        #velocity_t0 = velocity_t0.unsqueeze(0).repeat(j0.size()[0],1,1)
        #velocity_t0 = j0
        '''
        if clip_num == 0:
            #velocity_t0 = torch.zeros(seq_jacobians.size())[0].unsqueeze(0).float().cuda().contiguous()
            jac_delta = target_j - j0 
            time_delta = times[-1] - times[0]
            velocity_t0 = torch.div(jac_delta,time_delta) #*(1.0/6.0)
            velocity_t0 = velocity_t0.unsqueeze(0)
        else:
            #prev_clip_j0 = poisson_solver.jacobians_from_vertices(previous_clip_first_shape.unsqueeze(0)).contiguous()
            previous_pred_jacobians = poisson_solver.jacobians_from_vertices(previous_pred_seq)
            previous_velocities = self.get_gt_velocity(previous_pred_jacobians,previous_times)
            velocity_t0 = previous_velocities[-1].unsqueeze(0).clone().detach().contiguous()
        '''
        jac_delta = target_j - j0 
        time_delta = times[-1] - times[0]
        velocity_t0 = torch.div(jac_delta,time_delta)*(1.0/6) #*(1.0/2.0)
        velocity_t0 = velocity_t0.unsqueeze(0)

        #velocity_t0 = seq_jacobians[0].unsqueeze(0)
        if train:
            #jac_first_order = self.get_ode_solution_first_order(j0,target_j,centroids_normals,seq_encoding,times,betas,spf,centroids_normals_cat)
            #fo_j1 = jac_first_order[-1]
            #fo_j0 = jac_first_order[0]
            #velocity_t0 = torch.div((fo_j1-fo_j0),times[-1]-times[0]).float().cuda()
            target_j = self.decoder_network(seq_encoding,first_j0,full_sequence_times[split_indices[clip_num+1]])

            jac_first_order,dj_dt,jac_second_order = self.get_ode_solution(j0,velocity_t0,target_j,centroids_normals,seq_encoding,times,betas,spf,centroids_normals_cat)
        else:
            jac_first_order,dj_dt,jac_second_order = self.get_ode_solution(j0,velocity_t0,target_j,centroids_normals,seq_encoding,times,betas,spf,centroids_normals_cat)
        pred_full_jacobians = jac_second_order
        pred_full_jacobians_first_order = jac_first_order
        #jac_second_order = 0.5*(jac_second_order+jac_first_order)
        if train:
            batch_xip1 = poisson_solver.solve_poisson(pred_full_jacobians) #.squeeze()
            batch_xip1_first_order = poisson_solver.solve_poisson(jac_first_order) #.squeeze()
            pred_beta = self.get_beta(pred_full_jacobians,centroids_normals_cat,times)
            #gt_betas = self.get_beta(seq_jacobians,centroids_normals_cat,times)
        else:
            #pred_beta = self.get_beta(pred_full_jacobians,centroids_normals_cat,times)
            #jac_first_order,dj_dt,jac_second_order = self.get_ode_solution(j0,velocity_t0,centroids_normals,target_j,seq_encoding,times,pred_beta,spf,centroids_normals_cat)

            #blending version next 2 lines uncomment
            #pred_full_jacobians = jac_second_order
            #blended_jacobians = self.blend_predicted_jacobians_with_target(pred_full_jacobians,target_jacobians,times)

            #batch_xip1 = test_poisson_solver.solve_poisson(jac_second_order) #.squeeze()
            #batch_xip1 = test_poisson_solver.solve_poisson(blended_jacobians) #.squeeze()
            batch_xip1 = test_poisson_solver.solve_poisson(pred_full_jacobians) #.squeeze()

        if train:
            deformed_shape,_,_ = shape_normalization_transforms_pytorch(batch_xip1)
        else:
            deformed_shape = batch_xip1
        #_mesh = trimesh.Trimesh(vertices=deformed_shape[0].squeeze().cpu().detach().numpy(),faces=test_faces.squeeze().cpu().detach().numpy(),process=False)
        #_mesh.export("test_bunny.ply")
        #exit()
        if train:
            deformed_shape_first_order,_,_ = shape_normalization_transforms_pytorch(batch_xip1_first_order)
            #target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            #gt_full_jacobians = poisson_solver.jacobians_from_vertices(target_shape).double()
            #gt_full_jacobians = self.blend_predicted_jacobians_with_target(gt_full_jacobians,target_jacobians,times)
            #target_shape = poisson_solver.solve_poisson(seq_jacobians)
            target_shape,_,_ = shape_normalization_transforms_pytorch(seq)
            gt_full_jacobians = seq_jacobians #.double()

            gt_ae_jacobians = seq_jacobians_for_transformer[:,[-1],:].view(1,faces.size()[0],3,3).squeeze()
            dj_dt_gt = self.get_gt_velocity(seq_jacobians,times)
            #autoencoder_loss = torch.mean(torch.linalg.matrix_norm(pred_j-gt_ae_jacobians))
            autoencoder_loss = loss_args.mse(target_j,gt_ae_jacobians)
            #autoencoder_loss = loss_args.mse(pred_j,target_jacobians[0])
            seq_loss_shape = loss_args.mse(deformed_shape,target_shape)
            seq_loss_shape_first_order = loss_args.mse(deformed_shape_first_order,target_shape)
            #jac_loss = torch.mean(torch.linalg.matrix_norm(pred_full_jacobians-gt_full_jacobians))
            jac_loss = loss_args.mse(pred_full_jacobians,gt_full_jacobians)
            #jac_first_order_loss = torch.mean(torch.linalg.matrix_norm(pred_full_jacobians_first_order-gt_full_jacobians))
            jac_first_order_loss = loss_args.mse(pred_full_jacobians_first_order,gt_full_jacobians)
            gt_betas = betas.unsqueeze(0).unsqueeze(0)
            #gt_betas = self.positional_encoding(gt_betas).squeeze()
            #gt_betas = self.positional_encoding(gt_betas).squeeze()
            gt_betas = self.normalize_one_minus_one(gt_betas)
            beta_loss = loss_args.mse(pred_beta.squeeze(),gt_betas)
            #jac_velocity_loss = torch.mean(torch.linalg.matrix_norm(dj_dt[:-1]-dj_dt_gt))
            jac_velocity_loss = loss_args.mse(dj_dt[:-1],dj_dt_gt)
            print(dj_dt[1,:5].view(1,-1).squeeze())
            print(dj_dt_gt[1,:5].view(1,-1).squeeze())
            print()
            #print(dj_dt[1][:10].view(-1,9))
            #print(dj_dt_gt[1][:10].view(-1,9))
            #print()
            #print()
            #jac_velocity_loss = loss_args.mse(dj_dt[:-1],dj_dt_gt)
            #jac_velocity_loss = l1loss(dj_dt[:-1],dj_dt_gt)
            #dj_dt_gt = torch.cat([velocity_t0,dj_dt_gt],0)
            #if clip_num == 0:
            #    v0 = torch.zeros(dj_dt_gt[0].size()).unsqueeze(0).float().cuda()
            #    dj_dt_gt = torch.cat([v0,dj_dt_gt])
            #else:
            #previous_gt_jacobians = poisson_solver.jacobians_from_vertices(previous_seq).contiguous()
            #previous_velocities = self.get_gt_velocity(previous_gt_jacobians,previous_times)
            #v0 = previous_velocities[-1].unsqueeze(0).contiguous()
            #jac_velocity_loss = l1loss(dj_dt[1:],dj_dt_gt)
            #gaussian_loss = loss_args.mse(seq_encoding,gt_gauss_code)

            return autoencoder_loss,seq_loss_shape,seq_loss_shape_first_order,jac_loss,jac_first_order_loss,jac_velocity_loss,beta_loss,seq_encoding,deformed_shape
        else:
            return deformed_shape

    def get_sub_id(self,sub):
        if sub=="50007":
            sub_id = 0
        elif sub=="50009":
            sub_id = 1
        elif sub=="50020":
            sub_id = 2
        elif sub=="50022":
            sub_id = 3
        elif sub=="50025":
            sub_id = 4
        elif sub=="50026":
            sub_id = 5
        elif sub=="50027":
            sub_id = 6
        elif sub=="50002":
            sub_id = 7
        elif sub=="50004":
            sub_id = 8
        else:
            sub_id = 0

        return sub_id
        
    def amass_one_epoch(self,optimizer,epochs,loss_args,training_args,test_args,data_args,epoch):
        self.seq_dataset.shuffle()
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        mean_seq_loss = 0.0
        mean_jac_loss = 0.0
        mean_jac_first_loss = 0.0
        mean_jac_velocity_loss = 0.0
        mean_autoencoder_loss = 0.0
        mean_gaussian_loss = 0.0
        mean_beta_loss = 0.0
        num_sequences = 0.0
        logs_per_seq = 10
        prev_log = timer()
        logger = logging.getLogger("Running Epoch "+str(epoch))
        optimizer.zero_grad()
        batch_codes = []
        batch = 4
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0 ; batch_jac_first_order_loss = 0.0 ; batch_jac_second_order_loss = 0.0 ; batch_jac_velocity_loss = 0.0 ; batch_ae_loss = 0.0 ; batch_gaussian_loss = 0.0
        batch_beta_loss = 0.0 ; batch_seq_loss_first_order = 0.0 ; batch_seq_loss_second_order = 0.0
        primary_batch_seq_loss = 0.0 ; primary_batch_jac_loss = 0.0 ; primary_batch_jac_first_order_loss = 0.0 ; primary_batch_jac_velocity_loss = 0.0 ; primary_batch_ae_loss = 0.0 ; primary_batch_gaussian_loss = 0.0
        primary_batch_beta_loss = 0.0

        seq_loss_dict = {}
        all_subjects = []
        for bix,data in enumerate(dataloader):
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,gender,_,name = data
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps)
            zero_betas = torch.zeros(betas.size()).float().cuda()
            root_zero = torch.zeros(root_orient.size()).float().cuda()
            body_zero = torch.zeros(pose_body.size()).float().cuda()
            hand_zero = torch.zeros(pose_hand.size()).float().cuda()
            zero_trans = torch.zeros(trans.size()).float().cuda()
            primary_mean_sequence,_ = get_sequence(root_orient,pose_body,pose_hand,trans,zero_betas,dmpls,gender,data_args.max_frames,use_dmpl=False)
            wodmpl_sequence,_ = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=False)
            onlydmpl_sequence,_ = get_sequence(root_orient,body_zero,hand_zero,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            primary_mean_sequence,wodmpl_sequence,onlydmpl_sequence,sequence = align_feet(primary_mean_sequence,wodmpl_sequence,onlydmpl_sequence,sequence)
            first_shape = sequence[0]
            wodmpl_first_shape = wodmpl_sequence[0]
            onlydmpl_first_shape = onlydmpl_sequence[0]
            primary_mean_first_shape = primary_mean_sequence[0]
            sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
            mean_seq_seq_loss = 0.0
            mean_seq_jac_loss = 0.0
            mean_seq_jac_first_loss = 0.0
            mean_seq_jac_second_loss = 0.0
            mean_seq_jac_velocity_loss = 0.0
            mean_seq_ae_loss = 0.0
            mean_seq_g_loss = 0.0
            mean_seq_beta_loss = 0.0
            #for shix in range(len(seq_per_body_shape)):
            clips,clip_times,seq_times,split_indices = get_atomic_clips(sequence,self.frame_length,fps)
            clip_poses,_,_ = split_by_indices(sequence_pose,seq_times,split_indices)
            wodmpl_clips,_,_ = split_by_indices(wodmpl_sequence,seq_times,split_indices)
            onlydmpl_clips,_,_ = split_by_indices(onlydmpl_sequence,seq_times,split_indices)
            primary_mean_clips,_,_ = split_by_indices(primary_mean_sequence,seq_times,split_indices)
            tot_dist = torch.mean(torch.sqrt(torch.sum((sequence[-1,:,:] - sequence[0,:,:])**2,dim=1)))
            tot_time = seq_times[-1] - seq_times[0]
            speed = tot_dist/tot_time
            #clips = get_clips(sequence,self.frame_length)
            indices = list(range(len(clips)))
            random.shuffle(indices)
            clips = [clips[i] for i in indices] 
            clip_times = [clip_times[i] for i in indices]
            clip_poses = [clip_poses[i] for i in indices]
            wodmpl_clips = [wodmpl_clips[i] for i in indices]
            onlydmpl_clips = [onlydmpl_clips[i] for i in indices]
            primary_mean_clips = [primary_mean_clips[i] for i in indices]
            log_every = ceil(len(clips)*1.0/logs_per_seq)
            clip_ix = 0
            gt_color = np.array([175,225,175],dtype=float)/255.0

            previous_seq = None
            previous_pred_seq = None
            previous_times = None
            start_frame = 0
            for i,(clip,wodmpl_clip,onlydmpl_clip,primary_mean_clip,times,pose) in enumerate(zip(clips,wodmpl_clips,onlydmpl_clips,primary_mean_clips,clip_times,clip_poses)):
                #display_utils.write_amass_sequence("./motion_1/","0_100",clip,faces,start_frame,gt_color,"src")
                #display_utils.write_amass_sequence("./dmpl_motion_1/","0_100",dmpl_clip,faces,start_frame,gt_color,"src")
                #display_utils.write_amass_sequence("./mean_motion_1/","0_100",primary_mean_clip,faces,start_frame,gt_color,"src")
                shape_seq = clip
                wodmpl_shape_seq = wodmpl_clip
                onlydmpl_shape_seq = onlydmpl_clip
                primary_mean_shape_seq = primary_mean_clip
                if i<len(clips)-1:
                    #last_shape = clips[i+1][0].unsqueeze(0)
                    last_shape = clip[-1].unsqueeze(0)
                    wodmpl_last_shape = wodmpl_clip[-1].unsqueeze(0)
                    onlydmpl_last_shape = onlydmpl_clip[-1].unsqueeze(0)
                    #primary_mean_last_shape = primary_mean_clips[i+1][0].unsqueeze(0)
                    primary_mean_last_shape = primary_mean_clip[-1].unsqueeze(0)
                else:
                    last_shape = clip[-1].unsqueeze(0)
                    wodmpl_last_shape = wodmpl_clip[-1].unsqueeze(0)
                    onlydmpl_last_shape = onlydmpl_clip[-1].unsqueeze(0)
                    primary_mean_last_shape = primary_mean_clip[-1].unsqueeze(0)
                #times = get_time_for_clip(clip)
                #primmean_autoencoder_loss,primmean_seq_loss,primmean_seq_loss_first_order,primmean_jac_loss,primmean_jac_first_order_loss,primmean_jac_velocity_loss,primmean_beta_loss,primmean_seq_encoding = self.one_sequence(primary_mean_shape_seq,faces,times,primary_mean_first_shape,loss_args,spf=1.0/fps.item(),last_shape=primary_mean_last_shape,pose=pose,betas=zero_betas[0,0,0,:],encoding=training_args.encoding)
                #wodmpl_autoencoder_loss,wodmpl_seq_loss,wodmpl_seq_loss_first_order,wodmpl_jac_loss,wodmpl_jac_first_order_loss,wodmpl_jac_velocity_loss,wodmpl_beta_loss,wodmpl_seq_encoding,_= self.one_sequence(wodmpl_shape_seq,faces,times,wodmpl_first_shape,loss_args,spf=1.0/fps.item(),last_shape=wodmpl_last_shape,pose=pose,betas=zero_betas[0,0,0,:],encoding=training_args.encoding,full_sequence_pose=sequence_pose,full_sequence_times=seq_times,clip_num=i,previous_seq=previous_seq,previous_pred_seq=previous_pred_seq,previous_times=previous_times)
                #onlydmpl_autoencoder_loss,onlydmpl_seq_loss,onlydmpl_seq_loss_first_order,onlydmpl_jac_loss,onlydmpl_jac_first_order_loss,onlydmpl_jac_velocity_loss,onlydmpl_beta_loss,wodmpl_seq_encoding = self.one_sequence(onlydmpl_shape_seq,faces,times,onlydmpl_first_shape,loss_args,spf=1.0/fps.item(),last_shape=onlydmpl_last_shape,pose=pose,betas=betas[0,0,0,:],encoding=training_args.encoding)

                #pseudo_betas = torch.zeros(betas.size()).float().cuda()
                #sub = name[0].split("_")[0]
                #sub_id = self.get_sub_id(sub)
                #pseudo_betas += sub_id

                autoencoder_loss,seq_loss,seq_loss_first_order,jac_loss,jac_first_order_loss,jac_velocity_loss,beta_loss,seq_encoding,pred_seq = self.one_sequence(shape_seq,faces,times,first_shape,loss_args,spf=1.0/fps.item(),last_shape=last_shape,pose=pose,betas=betas[0,0,0,:],encoding=training_args.encoding,full_sequence_pose=sequence_pose,full_sequence_times=seq_times,clip_num=i,previous_seq=previous_seq,previous_pred_seq=previous_pred_seq,previous_times=previous_times,split_indices=split_indices)

                batch_codes.append(seq_encoding)
                #loss = seq_loss + 0.1*jac_loss + 0.1*autoencoder_loss + 0*gaussian_loss
                #loss.backward()
                #optimizer.step()
                #batch_seq_loss += (seq_loss+primmean_seq_loss+wodmpl_seq_loss+onlydmpl_seq_loss)
                #batch_seq_loss += (seq_loss+wodmpl_seq_loss)
                batch_seq_loss += (seq_loss)
                #batch_seq_loss_first_order += (seq_loss_first_order+primmean_seq_loss_first_order+wodmpl_seq_loss_first_order+onlydmpl_seq_loss_first_order)
                #batch_seq_loss_first_order += (seq_loss_first_order+wodmpl_seq_loss_first_order)
                batch_seq_loss_first_order += (seq_loss_first_order)
                #batch_seq_loss += seq_loss
                #batch_seq_loss += ((seq_loss+primmean_seq_loss)/2.0)
                #batch_jac_loss += (jac_loss+primmean_jac_loss+wodmpl_jac_loss+onlydmpl_jac_loss)
                #batch_jac_loss += (jac_loss+wodmpl_jac_loss)
                batch_jac_loss += (jac_loss)
                #batch_jac_loss += jac_loss
                #batch_jac_loss += ((jac_loss+primmean_jac_loss)/2.0)
                #batch_jac_first_order_loss += (jac_first_order_loss+primmean_jac_first_order_loss+wodmpl_jac_first_order_loss+onlydmpl_jac_first_order_loss)
                #batch_jac_first_order_loss += (jac_first_order_loss+wodmpl_jac_first_order_loss)
                batch_jac_first_order_loss += (jac_first_order_loss)
                #batch_jac_first_order_loss += jac_first_order_loss
                #batch_jac_first_order_loss += ((jac_first_order_loss+primmean_jac_first_order_loss)/2.0)
                #batch_jac_velocity_loss += (jac_velocity_loss+primmean_jac_velocity_loss+wodmpl_jac_velocity_loss+onlydmpl_jac_velocity_loss)
                #batch_jac_velocity_loss += (jac_velocity_loss+wodmpl_jac_velocity_loss)
                batch_jac_velocity_loss += (jac_velocity_loss)
                #batch_jac_velocity_loss += jac_velocity_loss
                #batch_jac_velocity_loss += ((jac_velocity_loss+primmean_jac_velocity_loss)/2.0)
                #batch_ae_loss += (autoencoder_loss+wodmpl_autoencoder_loss)
                batch_ae_loss += (autoencoder_loss)
                #batch_beta_loss += (beta_loss+primmean_beta_loss+wodmpl_beta_loss+onlydmpl_beta_loss)
                #batch_beta_loss += (beta_loss+wodmpl_beta_loss)
                batch_beta_loss += (beta_loss)
                mean_seq_seq_loss += seq_loss.detach().item()
                mean_seq_jac_loss += jac_loss.detach().item()
                mean_seq_jac_first_loss += jac_first_order_loss.detach().item()
                mean_seq_ae_loss += autoencoder_loss.detach().item()
                mean_seq_jac_velocity_loss += jac_velocity_loss.detach().item()
                mean_seq_beta_loss += beta_loss.detach().item()
                #mean_seq_g_loss += gaussian_loss.detach().item()
                previous_seq = shape_seq 
                previous_pred_seq = pred_seq
                previous_times = times
                now = timer()

                if len(batch_codes)==batch:
                    batch_gaussian_loss = self.regloss(torch.cat(batch_codes,0))
                    batch_seq_loss,batch_seq_loss_first_order,batch_jac_loss,batch_jac_first_order_loss,batch_jac_velocity_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss = self.batch_backprop(optimizer,batch_seq_loss,batch_seq_loss_first_order,batch_jac_loss,batch_jac_first_order_loss,batch_jac_velocity_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss,training_args,epoch=epoch)
                    batch_codes = []

                #if now - prev_log >= 60:
                if clip_ix == 0:
                    message = colored("{0:s} Sequence {1:3d} of {2:3d}".format(name[0],bix,len(dataloader)),"magenta")
                    logger.info(message)
                    #message = colored("Estimated {0:3.3f} Actual {1:3.3f} Speed {2:3.3f}".format(vid_l[0].item(),vid_l[1].item(),speed.item()),"red")
                    #logger.info(message)
                if  clip_ix % log_every == 0:
                    message = colored("Seq Loss: {0:2.6f} Jac loss: {1:2.6f} Jac First loss: {2:2.6f} Velocity loss: {3:2.6f} Beta Loss: {4:2.6f} AE Loss: {5:2.6f} Clip: {6:4d} of {7:4d}"
                    .format(mean_seq_seq_loss/(clip_ix+1),mean_seq_jac_loss/(clip_ix+1),mean_seq_jac_first_loss/(clip_ix+1),mean_seq_jac_velocity_loss/(clip_ix+1),mean_seq_beta_loss/(clip_ix+1),mean_seq_ae_loss/(clip_ix+1),clip_ix,len(clips)),'cyan')
                    logger.info(message)

                    prev_log = now
                clip_ix +=1
                start_frame += len(clip)

            mean_seq_loss += (mean_seq_seq_loss/len(clips))
            mean_jac_loss += (mean_seq_jac_loss/len(clips))
            mean_jac_first_loss += (mean_seq_jac_first_loss/len(clips))
            mean_jac_velocity_loss += (mean_seq_jac_velocity_loss/len(clips))
            mean_autoencoder_loss += (mean_seq_ae_loss/len(clips))
            mean_gaussian_loss += (mean_seq_g_loss/len(clips))
            mean_beta_loss += (mean_seq_beta_loss/len(clips))
            seq_loss_dict[name[0]] = mean_seq_seq_loss/len(clips)
            num_sequences += 1.0

            if len(batch_codes) > 0:
                if len(batch_codes) > 1:
                    batch_gaussian_loss = self.regloss(torch.cat(batch_codes,0))
                    batch_seq_loss,batch_seq_loss_first_order,batch_jac_loss,batch_jac_first_order_loss,batch_jac_velocity_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss = self.batch_backprop(optimizer,batch_seq_loss,batch_seq_loss_first_order,batch_jac_loss,batch_jac_first_order_loss,batch_jac_velocity_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss,training_args,epoch=epoch)
                    batch_codes = []
                else:
                    optimizer.zero_grad() ; self.jacobian_first_order.zero_grad() ; self.acceleration_func.zero_grad() ; self.pointnet_network.zero_grad() ; self.transformer_network.zero_grad() ; self.projection_network.zero_grad()
                    self.beta_network.zero_grad()

        for k,v in seq_loss_dict.items():
            message = colored("{0:s} : {1:2.6f}".format(k,v),"yellow")
            logger.info(message)
        mean_seq_loss /= num_sequences            
        mean_jac_loss /= num_sequences
        mean_jac_first_loss /= num_sequences
        mean_jac_velocity_loss /= num_sequences
        mean_autoencoder_loss /= num_sequences
        mean_gaussian_loss /= num_sequences
        mean_beta_loss /= num_sequences

        return mean_seq_loss,mean_jac_loss,mean_jac_first_loss,mean_jac_velocity_loss,mean_beta_loss,mean_autoencoder_loss,mean_gaussian_loss
        
    def amass_one_epoch_first_order(self,optimizer,epochs,loss_args,training_args,test_args,data_args,epoch):
        #self.seq_dataset.shuffle()
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        mean_seq_loss = 0.0
        mean_jac_loss = 0.0
        mean_jac_first_loss = 0.0
        mean_jac_velocity_loss = 0.0
        mean_autoencoder_loss = 0.0
        mean_gaussian_loss = 0.0
        mean_beta_loss = 0.0
        num_sequences = 0.0
        logs_per_seq = 10
        prev_log = timer()
        logger = logging.getLogger("Running Epoch "+str(epoch))
        optimizer.zero_grad()
        batch_codes = []
        batch = 4
        batch_seq_loss = 0.0 ; batch_jac_loss = 0.0 ; batch_jac_first_order_loss = 0.0 ; batch_jac_second_order_loss = 0.0 ; batch_jac_velocity_loss = 0.0 ; batch_ae_loss = 0.0 ; batch_gaussian_loss = 0.0
        batch_beta_loss = 0.0 ; batch_seq_loss_first_order = 0.0 ; batch_seq_loss_second_order = 0.0
        primary_batch_seq_loss = 0.0 ; primary_batch_jac_loss = 0.0 ; primary_batch_jac_first_order_loss = 0.0 ; primary_batch_jac_velocity_loss = 0.0 ; primary_batch_ae_loss = 0.0 ; primary_batch_gaussian_loss = 0.0
        primary_batch_beta_loss = 0.0

        seq_loss_dict = {}
        all_subjects = []
        for bix,data in enumerate(dataloader):
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,gender,_,name = data
            root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps)
            zero_betas = torch.zeros(betas.size()).float().cuda()
            root_zero = torch.zeros(root_orient.size()).float().cuda()
            body_zero = torch.zeros(pose_body.size()).float().cuda()
            hand_zero = torch.zeros(pose_hand.size()).float().cuda()
            zero_trans = torch.zeros(trans.size()).float().cuda()
            primary_mean_sequence,_ = get_sequence(root_orient,pose_body,pose_hand,trans,zero_betas,dmpls,gender,data_args.max_frames,use_dmpl=False)
            wodmpl_sequence,_ = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=False)
            onlydmpl_sequence,_ = get_sequence(root_orient,body_zero,hand_zero,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames,use_dmpl=True)
            primary_mean_sequence,wodmpl_sequence,onlydmpl_sequence,sequence = align_feet(primary_mean_sequence,wodmpl_sequence,onlydmpl_sequence,sequence)
            first_shape = sequence[0]
            wodmpl_first_shape = wodmpl_sequence[0]
            onlydmpl_first_shape = onlydmpl_sequence[0]
            primary_mean_first_shape = primary_mean_sequence[0]
            sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
            mean_seq_seq_loss = 0.0
            mean_seq_jac_loss = 0.0
            mean_seq_jac_first_loss = 0.0
            mean_seq_jac_second_loss = 0.0
            mean_seq_jac_velocity_loss = 0.0
            mean_seq_ae_loss = 0.0
            mean_seq_g_loss = 0.0
            mean_seq_beta_loss = 0.0
            #for shix in range(len(seq_per_body_shape)):
            clips,clip_times,seq_times,split_indices = get_atomic_clips(sequence,self.frame_length,fps)
            clip_poses,_,_ = split_by_indices(sequence_pose,seq_times,split_indices)
            wodmpl_clips,_,_ = split_by_indices(wodmpl_sequence,seq_times,split_indices)
            onlydmpl_clips,_,_ = split_by_indices(onlydmpl_sequence,seq_times,split_indices)
            primary_mean_clips,_,_ = split_by_indices(primary_mean_sequence,seq_times,split_indices)
            tot_dist = torch.mean(torch.sqrt(torch.sum((sequence[-1,:,:] - sequence[0,:,:])**2,dim=1)))
            tot_time = seq_times[-1] - seq_times[0]
            speed = tot_dist/tot_time
            #clips = get_clips(sequence,self.frame_length)
            indices = list(range(len(clips)))
            random.shuffle(indices)
            clips = [clips[i] for i in indices] 
            clip_times = [clip_times[i] for i in indices]
            clip_poses = [clip_poses[i] for i in indices]
            wodmpl_clips = [wodmpl_clips[i] for i in indices]
            onlydmpl_clips = [onlydmpl_clips[i] for i in indices]
            primary_mean_clips = [primary_mean_clips[i] for i in indices]
            log_every = ceil(len(clips)*1.0/logs_per_seq)
            clip_ix = 0
            gt_color = np.array([175,225,175],dtype=float)/255.0

            previous_seq = None
            previous_pred_seq = None
            previous_times = None
            start_frame = 0
            for i,(clip,wodmpl_clip,onlydmpl_clip,primary_mean_clip,times,pose) in enumerate(zip(clips,wodmpl_clips,onlydmpl_clips,primary_mean_clips,clip_times,clip_poses)):
                #display_utils.write_amass_sequence("./motion_1/","0_100",clip,faces,start_frame,gt_color,"src")
                #display_utils.write_amass_sequence("./dmpl_motion_1/","0_100",dmpl_clip,faces,start_frame,gt_color,"src")
                #display_utils.write_amass_sequence("./mean_motion_1/","0_100",primary_mean_clip,faces,start_frame,gt_color,"src")
                shape_seq = clip
                wodmpl_shape_seq = wodmpl_clip
                onlydmpl_shape_seq = onlydmpl_clip
                primary_mean_shape_seq = primary_mean_clip
                if i<len(clips)-1:
                    #last_shape = clips[i+1][0].unsqueeze(0)
                    last_shape = clip[-1].unsqueeze(0)
                else:
                    last_shape = clip[-1].unsqueeze(0)

                seq_loss,jac_loss,jac_velocity_loss,beta_loss,autoencoder_loss,seq_encoding,pred_seq = self.one_sequence_first_order(shape_seq,faces,times,first_shape,loss_args,spf=1.0/fps.item(),last_shape=last_shape,pose=pose,betas=betas[0,0,0,:],encoding=training_args.encoding,full_sequence_pose=sequence_pose,full_sequence_times=seq_times,clip_num=i,previous_seq=previous_seq,previous_pred_seq=previous_pred_seq,previous_times=previous_times,split_indices=split_indices)

                batch_codes.append(seq_encoding)
                batch_seq_loss += (seq_loss)
                batch_jac_loss += (jac_loss)
                batch_jac_velocity_loss += (jac_velocity_loss)
                batch_ae_loss += (autoencoder_loss)

                batch_beta_loss += (beta_loss)
                mean_seq_seq_loss += seq_loss.detach().item()
                mean_seq_jac_loss += jac_loss.detach().item()
                mean_seq_jac_velocity_loss += jac_velocity_loss.detach().item()
                mean_seq_beta_loss += beta_loss.detach().item()
                mean_seq_ae_loss += autoencoder_loss.detach().item()
                #mean_seq_g_loss += gaussian_loss.detach().item()
                previous_seq = shape_seq 
                previous_pred_seq = pred_seq
                previous_times = times
                now = timer()

                if len(batch_codes)==batch:
                    batch_gaussian_loss = self.regloss(torch.cat(batch_codes,0))
                    batch_seq_loss,batch_jac_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss = self.batch_backprop_first_order(optimizer,batch_seq_loss,batch_jac_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss,training_args,epoch=epoch)
                    batch_codes = []

                #if now - prev_log >= 60:
                if clip_ix == 0:
                    message = colored("{0:s} Sequence {1:3d} of {2:3d}".format(name[0],bix,len(dataloader)),"magenta")
                    logger.info(message)
                    #message = colored("Estimated {0:3.3f} Actual {1:3.3f} Speed {2:3.3f}".format(vid_l[0].item(),vid_l[1].item(),speed.item()),"red")
                    #logger.info(message)
                if  clip_ix % log_every == 0:
                    message = colored("Seq Loss: {0:2.6f} Jac loss: {1:2.6f} Velocity loss: {2:2.6f} Beta Loss: {3:2.6f} AE Loss:{4:2.6f} Clip: {5:4d} of {6:4d}"
                    .format(mean_seq_seq_loss/(clip_ix+1),mean_seq_jac_loss/(clip_ix+1),mean_seq_jac_velocity_loss/(clip_ix+1),mean_seq_beta_loss/(clip_ix+1),mean_seq_ae_loss/(clip_ix+1),clip_ix,len(clips)),'cyan')
                    logger.info(message)

                    prev_log = now
                clip_ix +=1
                start_frame += len(clip)

            mean_seq_loss += (mean_seq_seq_loss/len(clips))
            mean_jac_loss += (mean_seq_jac_loss/len(clips))
            mean_jac_first_loss += (mean_seq_jac_first_loss/len(clips))
            mean_jac_velocity_loss += (mean_seq_jac_velocity_loss/len(clips))
            mean_autoencoder_loss += (mean_seq_ae_loss/len(clips))
            mean_gaussian_loss += (mean_seq_g_loss/len(clips))
            mean_beta_loss += (mean_seq_beta_loss/len(clips))
            seq_loss_dict[name[0]] = mean_seq_seq_loss/len(clips)
            num_sequences += 1.0

            if len(batch_codes) > 0:
                if len(batch_codes) > 1:
                    batch_gaussian_loss = self.regloss(torch.cat(batch_codes,0))
                    batch_seq_loss,batch_jac_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss = self.batch_backprop_first_order(optimizer,batch_seq_loss,batch_jac_loss,batch_beta_loss,batch_ae_loss,batch_gaussian_loss,training_args,epoch=epoch)
                    batch_codes = []
                else:
                    optimizer.zero_grad() ; self.jacobian_first_order.zero_grad() ; self.acceleration_func.zero_grad() ; self.pointnet_network.zero_grad() ; self.transformer_network.zero_grad() ; self.projection_network.zero_grad()
                    self.beta_network.zero_grad()

        for k,v in seq_loss_dict.items():
            message = colored("{0:s} : {1:2.6f}".format(k,v),"yellow")
            logger.info(message)
        mean_seq_loss /= num_sequences            
        mean_jac_loss /= num_sequences
        mean_jac_first_loss /= num_sequences
        mean_jac_velocity_loss /= num_sequences
        mean_autoencoder_loss /= num_sequences
        mean_gaussian_loss /= num_sequences
        mean_beta_loss /= num_sequences

        return mean_seq_loss,mean_jac_loss,mean_jac_velocity_loss,mean_beta_loss,mean_autoencoder_loss,mean_gaussian_loss

    def loop_epochs(self,optimizer,epochs,loss_args,training_args,test_args,data_args):
        scheduler = CosineAnnealingLR(optimizer,
                                    T_max = 300, # Maximum number of iterations.
                                    eta_min = 1e-4) # Minimum learning rate.

        best_seq_loss = 1e9
        best_jac_loss = 1e9
        best_jac_velocity_loss = 1e9
        best_ae_loss = 1e9
        best_gaussian_loss = 1e9
        best_jac_first_loss = 1e9
        best_beta_loss = 1e9
        #self.pointnet_network.load_state_dict(torch.load(training_args.weight_path+'_pointnet'))
        #self.jacobian_first_order.load_state_dict(torch.load(training_args.weight_path+'_tangent'))
        #self.transformer_network.load_state_dict(torch.load(training_args.weight_path+'_transformer'))
        #self.acceleration_func.load_state_dict(torch.load(training_args.weight_path+'_acceleration'))
        #self.beta_network.load_state_dict(torch.load(training_args.weight_path+'_beta'))
        #self.decoder_network.load_state_dict(torch.load(training_args.weight_path+'_decoder_fo'))
        for i in range(0):
            logger = logging.getLogger("Finished Epoch "+str(i))
            ep_seq_loss,ep_jac_loss,ep_jac_velocity_loss,ep_beta_loss,ep_autoencoder_loss,ep_gaussian_loss = self.amass_one_epoch_first_order(optimizer,epochs,loss_args,training_args,test_args,data_args,i)
            #scheduler.step()
            if ep_seq_loss < best_seq_loss:
                best_seq_loss = ep_seq_loss
                best_epoch = i
                torch.save(self.jacobian_first_order.state_dict(),training_args.weight_path+'_tangent_fo')
                torch.save(self.pointnet_network.state_dict(),training_args.weight_path+'_pointnet_fo')
                torch.save(self.transformer_network.state_dict(),training_args.weight_path+'_transformer_fo')
                torch.save(self.acceleration_func.state_dict(),training_args.weight_path+'_acceleration_fo')
                torch.save(self.beta_network.state_dict(),training_args.weight_path+'_beta_fo')
                torch.save(self.projection_network.state_dict(),training_args.weight_path+'_projection_fo')
                torch.save(self.decoder_network.state_dict(),training_args.weight_path+'_decoder_fo')

            if ep_jac_loss < best_jac_loss:
                best_jac_loss = ep_jac_loss
            if ep_jac_velocity_loss < best_jac_velocity_loss:
                best_jac_velocity_loss = ep_jac_velocity_loss
            if ep_autoencoder_loss < best_ae_loss:
                best_ae_loss = ep_autoencoder_loss
            if ep_gaussian_loss < best_gaussian_loss:
                best_gaussian_loss = ep_gaussian_loss
            if ep_beta_loss < best_beta_loss:
                best_beta_loss = ep_beta_loss

            message = colored("Best Ep: {0:3d} Best Seq Loss: {1:2.6f} Best Jac loss: {2:2.6f} Best Jac First loss: {3:2.6f} Best Vel loss: {4:2.6f} Best Beta Loss: {5:2.6f} Best AE Loss: {6:2.6f} Best Gaussian Loss: {7:2.6f}"
            .format(best_epoch,best_seq_loss,best_jac_loss,best_jac_first_loss,best_jac_velocity_loss,best_beta_loss,best_ae_loss,best_gaussian_loss),'green')
            logger.info(message)
        for i in range(400):
            logger = logging.getLogger("Finished Epoch "+str(i))
            ep_seq_loss,ep_jac_loss,ep_jac_first_loss,ep_jac_velocity_loss,ep_beta_loss,ep_autoencoder_loss,ep_gaussian_loss = self.amass_one_epoch(optimizer,epochs,loss_args,training_args,test_args,data_args,i)
            #scheduler.step()
            if ep_seq_loss < best_seq_loss:
                best_seq_loss = ep_seq_loss
                best_epoch = i
                torch.save(self.jacobian_first_order.state_dict(),training_args.weight_path+'_tangent')
                torch.save(self.pointnet_network.state_dict(),training_args.weight_path+'_pointnet')
                torch.save(self.transformer_network.state_dict(),training_args.weight_path+'_transformer')
                torch.save(self.acceleration_func.state_dict(),training_args.weight_path+'_acceleration')
                torch.save(self.beta_network.state_dict(),training_args.weight_path+'_beta')
                torch.save(self.projection_network.state_dict(),training_args.weight_path+'_projection')
                torch.save(self.decoder_network.state_dict(),training_args.weight_path+'_decoder')

            if ep_jac_loss < best_jac_loss:
                best_jac_loss = ep_jac_loss
            if ep_jac_velocity_loss < best_jac_velocity_loss:
                best_jac_velocity_loss = ep_jac_velocity_loss
            if ep_autoencoder_loss < best_ae_loss:
                best_ae_loss = ep_autoencoder_loss
            if ep_gaussian_loss < best_gaussian_loss:
                best_gaussian_loss = ep_gaussian_loss
            if ep_jac_first_loss < best_jac_first_loss:
                best_jac_first_loss = ep_jac_first_loss
            if ep_beta_loss < best_beta_loss:
                best_beta_loss = ep_beta_loss

            message = colored("Best Ep: {0:3d} Best Seq Loss: {1:2.6f} Best Jac loss: {2:2.6f} Best Jac First loss: {3:2.6f} Best Vel loss: {4:2.6f} Best Beta Loss: {5:2.6f} Best AE Loss: {6:2.6f} Best Gaussian Loss: {7:2.6f}"
            .format(best_epoch,best_seq_loss,best_jac_loss,best_jac_first_loss,best_jac_velocity_loss,best_beta_loss,best_ae_loss,best_gaussian_loss),'green')
            logger.info(message)

    def def_trans_multiple_first_order(self,optimizer,epochs,loss_args,training_args,test_args,data_args,method=""):
        self.pointnet_network.load_state_dict(torch.load(training_args.weight_path+'_pointnet'))
        self.jacobian_first_order.load_state_dict(torch.load(training_args.weight_path+'_tangent'))
        self.transformer_network.load_state_dict(torch.load(training_args.weight_path+'_transformer'))
        self.acceleration_func.load_state_dict(torch.load(training_args.weight_path+'_acceleration'))
        self.beta_network.load_state_dict(torch.load(training_args.weight_path+'_beta'))
        self.projection_network.load_state_dict(torch.load(training_args.weight_path+'_projection'))
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)
        test_dataloader = DataLoader(self.test_seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        transfer_color = np.array([64,224,208],dtype=float)/255.0
        source_color = np.array([95,158,160],dtype=float)/255.0
        gt_color = np.array([175,225,175],dtype=float)/255.0
        root_dir = test_args.int_output_dir

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
                    source_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames)
                    mean_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,zero_betas,dmpls,gender,data_args.max_frames,use_dmpl=False)
                    mean_sequence_np = mean_sequence.cpu().detach().numpy()
                    if test_args.sig:
                        sk_vertices,sk_faces = sequence_to_skeleton_sequence(mean_sequence_np,faces.cpu().detach().numpy())
                    sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                    test_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames)
                    test_sequence_wodmpl,_ = get_sequence(root_orient,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=False)
                    #source_sequence,test_sequence_wodmpl,test_sequence = align_feet(source_sequence,test_sequence_wodmpl,test_sequence)
                    rest_shape = test_sequence[0]
                    rest_shape_wodmpl = test_sequence_wodmpl[0]
                    source_clips,source_clip_times,sequence_times,split_indices = get_atomic_clips(source_sequence,self.frame_length,fps)
                    test_clips,_,_ = split_by_indices(test_sequence,sequence_times,split_indices)
                    test_clips_wodmpl,_,_ = split_by_indices(test_sequence_wodmpl,sequence_times,split_indices)
                    clip_poses,_,_ = split_by_indices(sequence_pose,sequence_times,split_indices)
                    if test_args.sig:
                        sk_clips,_,_ = split_by_indices(sk_vertices,sequence_times,split_indices)
                    #test_clips,_,_ = get_atomic_clips(test_sequence,self.frame_length,fps)

                    start_shape = test_clips[0][0].unsqueeze(0)
                    start_shape_wodmpl = test_clips_wodmpl[0][0].unsqueeze(0)
                    #start_shape = source_clips[0][0].unsqueeze(0)

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

                    seq_primary_errors = []
                    seq_total_errors = []
                    pred_seq = []
                    seq_gt_disps = []
                    seq_pred_disps = []
                    acc_seq_v2v_errors = []
                    total_movements = []
                    gt_sequence = []
                    for i,(clip,test_clip,test_clip_wodmpl,times,source_pose) in enumerate(zip(source_clips,test_clips,test_clips_wodmpl,source_clip_times,clip_poses)):
                        if i<len(source_clips)-1:
                            #last_shape = test_clips[i+1][0].unsqueeze(0)
                            last_shape = test_clip[-1].unsqueeze(0)
                            last_shape_wodmpl = test_clip_wodmpl[-1].unsqueeze(0)
                            #last_shape = source_clips[i+1][0].unsqueeze(0)
                        else:
                            last_shape = test_clip[-1].unsqueeze(0)
                            last_shape_wodmpl = test_clip_wodmpl[-1].unsqueeze(0)
                            #last_shape = clip[-1].unsqueeze(0)
                        start_shape = test_clip[0].unsqueeze(0)
                        if method=="eulernjf":
                            #pseudo_betas = torch.zeros(betas.size()).float().cuda()
                            #sub = test_name[0].split("_")[0]
                            #sub_id = self.get_sub_id(sub)
                            #pseudo_betas += sub_id

                            trans_sequence = self.one_sequence_first_order(clip,faces,times,rest_shape,loss_args,train=False,test_shape=start_shape,spf=1.0/fps.item(),last_shape=last_shape,pose=source_pose,betas=betas[0,0,0,:],encoding=training_args.encoding,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times)
                        v2v_errors,v2p_errors = compute_errors(trans_sequence,test_clip,faces)
                        acc_seq_v2v_errors.append(v2v_errors)
                        gt_displacement = soft_displacements(test_clip,test_clip_wodmpl)
                        seq_total_errors.append(v2v_errors)
                        seq_gt_disps.append(gt_displacement)
                        pred_seq.append(trans_sequence)
                        gt_sequence.append(test_clip)
                        #soft_v2v = torch.absolute(v2v_errors - soft_v2v)
                        #soft_v2v = torch.div(primary_v2v_errors,soft_v2v)
                        all_v2v_errors.append(torch.mean(v2v_errors).item())
                        all_v2p_errors.append(torch.mean(v2p_errors).item())
                        ####start_shape = trans_sequence[-1].float().unsqueeze(0)
                        frame_ix += len(clip)
                        seq_v2p_error += torch.mean(v2p_errors).item()
                        seq_v2v_error += torch.mean(v2v_errors).item()

                        seq_name = str(start_frame) + "_" + str(start_frame+len(clip))
                        out_dir = os.path.join(root_dir,name,test_args.baseline)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        clip_ix += 1
                        message = colored("Rendered {0:3d} of {1:3d}".format(clip_ix,len(source_clips)),'blue')
                        logger.info(message)


                    seq_total_errors = torch.cat(seq_total_errors,0)
                    acc_seq_v2v_errors = torch.cat(acc_seq_v2v_errors,0)

                    per_vertex_acc_errors = acc_seq_v2v_errors.clone().detach()
                    per_vertex_acc_errors = torch.cumsum(per_vertex_acc_errors,dim=0)
                    frame_steps = torch.arange(0,per_vertex_acc_errors.size()[0],1).unsqueeze(-1).repeat(1,per_vertex_acc_errors.size()[1]).float().cuda()
                    frame_steps += 1
                    per_vertex_acc_errors = torch.div(per_vertex_acc_errors,frame_steps)

                    acc_seq_v2v_errors = torch.mean(acc_seq_v2v_errors,dim=-1)
                    acc_seq_v2v_errors = torch.cumsum(acc_seq_v2v_errors,dim=0).unsqueeze(-1).cpu().detach().numpy().tolist()
                    cum_out_dir = os.path.join(root_dir,"cumulative_"+test_args.baseline)
                    if not os.path.exists(cum_out_dir):
                        os.makedirs(cum_out_dir)
                    acc_file_name = os.path.join(cum_out_dir,name+".csv")
                    with open(acc_file_name,"w") as f:
                        csv.writer(f,delimiter='\n').writerows(acc_seq_v2v_errors)

                    start_frame = 0
                    if True: #test_args.sig:
                        #ref_feet,_ = torch.min(sk_vertices[0],dim=0,keepdim=True)
                        #pred_feet,_ = torch.min(cat_pred_seq[0],dim=0,keepdim=True)
                        #translation = (ref_feet[:,2] - pred_feet[:,2]).unsqueeze(0)
                        #for i,(trans_sequence,sk_clip) in enumerate(zip(pred_seq,sk_clips)):
                        for i,trans_sequence in enumerate(pred_seq):
                            #trans_sequence[:,:,2] += translation
                            #display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt",soft_errors=soft_v2v[start_frame:start_frame+len(trans_sequence)]) #v2v_errors=per_vertex_acc_errors[start_frame:start_frame+len(trans_sequence)])
                            display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt")
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
            with open(os.path.join(root_dir,test_args.baseline.strip("/")+"_mean.csv"),"w") as f:
                csv.writer(f,delimiter=" ").writerows([["v2p:",np.mean(all_v2p_errors),"v2v:",np.mean(all_v2v_errors)]])
            with open(os.path.join(root_dir,test_args.baseline.strip("/")+"_v2p.csv"),"w") as f:
                csv.writer(f,delimiter=",").writerows(name_and_v2p_errors)
            with open(os.path.join(root_dir,test_args.baseline.strip("/")+"_v2v.csv"),"w") as f:
                csv.writer(f,delimiter=",").writerows(name_and_v2v_errors)

    def def_trans_multiple(self,optimizer,epochs,loss_args,training_args,test_args,data_args,method=""):
        self.pointnet_network.load_state_dict(torch.load(training_args.weight_path+'_pointnet'))
        self.jacobian_first_order.load_state_dict(torch.load(training_args.weight_path+'_tangent'))
        self.transformer_network.load_state_dict(torch.load(training_args.weight_path+'_transformer'))
        self.acceleration_func.load_state_dict(torch.load(training_args.weight_path+'_acceleration'))
        self.beta_network.load_state_dict(torch.load(training_args.weight_path+'_beta'))
        self.projection_network.load_state_dict(torch.load(training_args.weight_path+'_projection'))
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)
        test_dataloader = DataLoader(self.test_seq_dataset,training_args.batch,shuffle=False,num_workers=1)

        transfer_color = np.array([64,224,208],dtype=float)/255.0
        source_color = np.array([95,158,160],dtype=float)/255.0
        gt_color = np.array([175,225,175],dtype=float)/255.0
        root_dir = test_args.int_output_dir

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
                    source_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames)
                    mean_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,zero_betas,dmpls,gender,data_args.max_frames,use_dmpl=False)
                    mean_sequence_np = mean_sequence.cpu().detach().numpy()
                    if test_args.sig:
                        sk_vertices,sk_faces = sequence_to_skeleton_sequence(mean_sequence_np,faces.cpu().detach().numpy())
                    sequence_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                    test_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames)
                    test_sequence_wodmpl,_ = get_sequence(root_orient,pose_body,pose_hand,trans,test_betas,dmpls,test_gender,data_args.max_frames,use_dmpl=False)
                    #source_sequence,test_sequence_wodmpl,test_sequence = align_feet(source_sequence,test_sequence_wodmpl,test_sequence)
                    rest_shape = test_sequence[0]
                    rest_shape_wodmpl = test_sequence_wodmpl[0]
                    source_clips,source_clip_times,sequence_times,split_indices = get_atomic_clips(source_sequence,self.frame_length,fps)
                    test_clips,_,_ = split_by_indices(test_sequence,sequence_times,split_indices)
                    test_clips_wodmpl,_,_ = split_by_indices(test_sequence_wodmpl,sequence_times,split_indices)
                    clip_poses,_,_ = split_by_indices(sequence_pose,sequence_times,split_indices)
                    if test_args.sig:
                        sk_clips,_,_ = split_by_indices(sk_vertices,sequence_times,split_indices)
                    #test_clips,_,_ = get_atomic_clips(test_sequence,self.frame_length,fps)

                    start_shape = test_clips[0][0].unsqueeze(0)
                    start_shape_wodmpl = test_clips_wodmpl[0][0].unsqueeze(0)
                    #start_shape = source_clips[0][0].unsqueeze(0)

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

                    seq_primary_errors = []
                    seq_total_errors = []
                    pred_seq = []
                    seq_gt_disps = []
                    seq_pred_disps = []
                    acc_seq_v2v_errors = []
                    total_movements = []
                    gt_sequence = []
                    for i,(clip,test_clip,test_clip_wodmpl,times,source_pose) in enumerate(zip(source_clips,test_clips,test_clips_wodmpl,source_clip_times,clip_poses)):
                        if i<len(source_clips)-1:
                            #last_shape = test_clips[i+1][0].unsqueeze(0)
                            last_shape = test_clip[-1].unsqueeze(0)
                            last_shape_wodmpl = test_clip_wodmpl[-1].unsqueeze(0)
                            #last_shape = source_clips[i+1][0].unsqueeze(0)
                        else:
                            last_shape = test_clip[-1].unsqueeze(0)
                            last_shape_wodmpl = test_clip_wodmpl[-1].unsqueeze(0)
                            #last_shape = clip[-1].unsqueeze(0)
                        start_shape = test_clip[0].unsqueeze(0)
                        if method=="eulernjf":
                            #pseudo_betas = torch.zeros(betas.size()).float().cuda()
                            #sub = test_name[0].split("_")[0]
                            #sub_id = self.get_sub_id(sub)
                            #pseudo_betas += sub_id

                            trans_sequence = self.one_sequence(clip,faces,times,rest_shape,loss_args,train=False,test_shape=start_shape,spf=1.0/fps.item(),last_shape=last_shape,pose=source_pose,betas=betas[0,0,0,:],encoding=training_args.encoding,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times)
                            trans_sequence_wodmpl = self.one_sequence(clip,faces,times,rest_shape_wodmpl,loss_args,train=False,test_shape=start_shape_wodmpl,spf=1.0/fps.item(),last_shape=last_shape_wodmpl,pose=source_pose,betas=zero_betas[0,0,0,:],encoding=training_args.encoding,full_sequence_pose=sequence_pose,full_sequence_times=sequence_times)
                        v2v_errors,v2p_errors = compute_errors(trans_sequence,test_clip,faces)
                        acc_seq_v2v_errors.append(v2v_errors)
                        primary_v2v_errors,primary_v2p_errors = compute_errors(trans_sequence,test_clip_wodmpl,faces)
                        #gt_displacement,_ = compute_errors(test_clip,test_clip_wodmpl,faces)
                        #pred_displacement,_ = compute_errors(trans_sequence,trans_sequence_wodmpl,faces)
                        pred_displacement = soft_displacements(trans_sequence,trans_sequence_wodmpl)
                        gt_displacement = soft_displacements(test_clip,test_clip_wodmpl)
                        #soft_v2v = torch.exp(-1*(primary_v2p_errors-v2p_errors))
                        #mean_motion = torch.mean(torch.sqrt(torch.sum((test_clip[1:]-test_clip[:-1])**2,dim=-1)),dim=0,keepdims=True)
                        #gt_disps = torch.div(gt_displacement,mean_motion)
                        #soft_v2v = torch.div(v2v_errors,gt_disps)
                        #soft_v2v = 1.0-((soft_v2v-torch.min(soft_v2v))/(torch.max(soft_v2v)-torch.min(soft_v2v)))
                        #soft_v2v = torch.div(primary_v2v_errors,v2v_errors)
                        #soft_v2v = compute_soft(trans_sequence,trans_sequence_wodmpl)
                        #soft_v2v = torch.absolute(trans_sequence - trans_sequence_wodmpl)
                        seq_primary_errors.append(primary_v2v_errors)
                        seq_total_errors.append(v2v_errors)
                        seq_gt_disps.append(gt_displacement)
                        seq_pred_disps.append(pred_displacement)
                        pred_seq.append(trans_sequence)
                        gt_sequence.append(test_clip)
                        #soft_v2v = torch.absolute(v2v_errors - soft_v2v)
                        #soft_v2v = torch.div(primary_v2v_errors,soft_v2v)
                        all_v2v_errors.append(torch.mean(v2v_errors).item())
                        all_v2p_errors.append(torch.mean(v2p_errors).item())
                        ####start_shape = trans_sequence[-1].float().unsqueeze(0)
                        start_shape_wodmpl = trans_sequence_wodmpl[-1].float().unsqueeze(0)
                        frame_ix += len(clip)
                        seq_v2p_error += torch.mean(v2p_errors).item()
                        seq_v2v_error += torch.mean(v2v_errors).item()

                        seq_name = str(start_frame) + "_" + str(start_frame+len(clip))
                        out_dir = os.path.join(root_dir,name,test_args.baseline)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        #display_utils.write_amass_sequence(out_dir,seq_name,clip,faces,start_frame,source_color,"src")
                        #display_utils.write_amass_sequence(out_dir,seq_name,sk_clip,sk_faces,start_frame,source_color,"src")
                        #display_utils.write_amass_sequence(out_dir,seq_name,trans_sequence,faces,start_frame,transfer_color,"tgt") #,soft_errors=soft_v2v)
                        #display_utils.write_amass_sequence(out_dir,seq_name,test_clip,faces,start_frame,gt_color,"gt")
                        #display_utils.write_amass_sequence(out_dir,seq_name,trans_sequence_wodmpl,faces,start_frame,transfer_color,"wodmpl")
                        #display_utils.write_amass_sequence(out_dir,seq_name,trans_sequence,faces,start_frame,transfer_color,"v2v",v2v_errors=v2v_errors)
                        #display_utils.write_amass_sequence(out_dir,seq_name,trans_sequence,faces,start_frame,transfer_color,"v2p",v2p_errors=v2p_errors)
                        start_frame += len(clip) 
                        clip_ix += 1
                        message = colored("Rendered {0:3d} of {1:3d}".format(clip_ix,len(source_clips)),'blue')
                        logger.info(message)


                    seq_total_errors = torch.cat(seq_total_errors,0)
                    seq_primary_errors = torch.cat(seq_primary_errors,0)
                    acc_seq_v2v_errors = torch.cat(acc_seq_v2v_errors,0)
                    seq_pred_disps = torch.linalg.norm(torch.cat(seq_pred_disps,0),dim=-1)
                    seq_gt_disps = torch.cat(seq_gt_disps,0)
                    gt_disp_norm = torch.linalg.norm(seq_gt_disps,dim=-1)
                    gt_seq = torch.cat(gt_sequence,0)
                    #mean_motion = torch.mean(torch.sqrt(torch.sum((gt_seq[1:]-gt_seq[:-1])**2,dim=-1)),dim=0,keepdims=True)
                    cat_pred_seq = torch.cat(pred_seq,0)
                    mean_motion = torch.mean(torch.sqrt(torch.sum((cat_pred_seq[1:]-cat_pred_seq[:-1])**2,dim=-1)),dim=0) #,keepdims=True)
                    #zero_disp = torch.zeros(1,mean_motion.size()[1]).float().cuda()
                    #mean_motion = torch.cat([zero_disp,mean_motion],0)
                    #total_motion = torch.cumsum(torch.linalg.norm(gt_seq - gt_seq[0],dim=-1),dim=0)
                    #seq_gt_disps = torch.div(seq_gt_disps,gt_disp_norm)
                    #print(seq_gt_disps.size())
                    #exit()

                    per_vertex_acc_errors = acc_seq_v2v_errors.clone().detach()
                    per_vertex_acc_errors = torch.cumsum(per_vertex_acc_errors,dim=0)
                    frame_steps = torch.arange(0,per_vertex_acc_errors.size()[0],1).unsqueeze(-1).repeat(1,per_vertex_acc_errors.size()[1]).float().cuda()
                    frame_steps += 1
                    per_vertex_acc_errors = torch.div(per_vertex_acc_errors,frame_steps)

                    acc_seq_v2v_errors = torch.mean(acc_seq_v2v_errors,dim=-1)
                    acc_seq_v2v_errors = torch.cumsum(acc_seq_v2v_errors,dim=0).unsqueeze(-1).cpu().detach().numpy().tolist()
                    cum_out_dir = os.path.join(root_dir,"cumulative_"+test_args.baseline)
                    if not os.path.exists(cum_out_dir):
                        os.makedirs(cum_out_dir)
                    acc_file_name = os.path.join(cum_out_dir,name+".csv")
                    with open(acc_file_name,"w") as f:
                        csv.writer(f,delimiter='\n').writerows(acc_seq_v2v_errors)
                    '''
                    working
                    seq_gt_disps = torch.cat(seq_gt_disps,0)
                    max_gt_disps,_ = torch.max(seq_gt_disps,dim=1,keepdims=True)
                    max_gt_disps = max_gt_disps.repeat(1,seq_gt_disps.size()[1])
                    '''
                    '''
                    seq_gt_disps = torch.cat(seq_gt_disps,0)
                    max_gt_disps,_ = torch.max(seq_gt_disps,dim=1,keepdims=True)
                    max_gt_disps = max_gt_disps.repeat(1,seq_gt_disps.size()[1])
                    seq_gt_disps = torch.div(seq_gt_disps,max_gt_disps)
                    '''

                    '''
                    cat_pred_seq = torch.cat(pred_seq,0)
                    mean_motion = torch.mean(torch.sqrt(torch.sum((cat_pred_seq[1:]-cat_pred_seq[:-1])**2,dim=-1)),dim=0,keepdims=True)
                    seq_gt_disps = torch.div(seq_gt_disps,mean_motion)
                    seq_gt_disps = (seq_gt_disps-torch.min(seq_gt_disps))/(torch.max(seq_gt_disps)-torch.min(seq_gt_disps))
                    #gt_disps = torch.div(max_gt_disps,mean_motion)
                    soft_v2v = torch.exp(-1*(seq_primary_errors-seq_total_errors))
                    soft_v2v = torch.div(soft_v2v,mean_motion)
                    '''
                    '''
                    working
                    soft_v2v = torch.div(seq_total_errors,max_gt_disps)
                    '''
                    '''
                    time_min,_ = torch.min(soft_v2v,dim=1,keepdims=True)
                    time_max,_ = torch.max(soft_v2v,dim=1,keepdims=True)
                    time_min = time_min.repeat(1,soft_v2v.size()[1])
                    time_max = time_max.repeat(1,soft_v2v.size()[1])
                    #soft_v2v = torch.div(soft_v2v,seq_gt_disps)
                    #soft_v2v = torch.div(soft_v2v,mean_motion)
                    soft_v2v = ((soft_v2v-time_min)/(time_max-time_min))
                    '''

                    #soft_v2v = torch.div(seq_pred_disps,mean_motion)
                    soft_v2v = seq_pred_disps
                    #print(seq_pred_disps.size(),mean_motion.size())
                    #soft_v2v = torch.div(seq_pred_disps,mean_motion)
                    #time_mean = torch.mean(soft_v2v,dim=0)
                    time_min,_ = torch.min(soft_v2v,dim=1,keepdims=True)
                    time_max,_ = torch.max(soft_v2v,dim=1,keepdims=True)
                    time_min = time_min.repeat(1,soft_v2v.size()[1])
                    time_max = time_max.repeat(1,soft_v2v.size()[1])
                    #mean_min = torch.min(time_mean)
                    #mean_max = torch.max(time_mean)
                    soft_v2v = ((soft_v2v-time_min)/(time_max-time_min))

                    #soft_v2v = ((time_mean-mean_min)/(mean_max-mean_min))
                    #soft_v2v = soft_v2v.repeat(gt_disp_norm.size()[0],1)

                    #time_min,_ = torch.min(seq_gt_disps,dim=1,keepdims=True)
                    #time_max,_ = torch.max(seq_gt_disps,dim=1,keepdims=True)
                    #time_min = time_min.repeat(1,seq_gt_disps.size()[1])
                    #time_max = time_max.repeat(1,seq_gt_disps.size()[1])
                    #seq_gt_disps = ((seq_gt_disps-time_min)/(time_max-time_min))

                    # might need later
                    start_frame = 0
                    per_time_acc_errors = torch.mean(per_vertex_acc_errors,dim=-1,keepdim=True)
                    frame_steps = torch.arange(0,per_vertex_acc_errors.size()[0],1).unsqueeze(-1).float().cuda()
                    per_time_acc_errors = torch.cat([frame_steps,per_time_acc_errors],-1).cpu().detach().numpy().tolist()
                    with open(os.path.join(root_dir,name,test_args.baseline.strip("/")+"_perframe.csv"),"w") as f:
                        csv.writer(f,delimiter=" ").writerows(per_time_acc_errors)
                    #for i,(trans_sequence,test_clip) in enumerate(zip(pred_seq,test_clips)):
                    #    print(i)
                    #    display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt",v2v_errors=per_vertex_acc_errors)
                    #    #display_utils.write_amass_sequence(out_dir,"",test_clip,faces,start_frame,gt_color,"gt",soft_errors=seq_gt_disps[start_frame:start_frame+len(test_clip)])
                    #    start_frame+=len(trans_sequence)
                    start_frame = 0
                    if True: #test_args.sig:
                        #ref_feet,_ = torch.min(sk_vertices[0],dim=0,keepdim=True)
                        #pred_feet,_ = torch.min(cat_pred_seq[0],dim=0,keepdim=True)
                        #translation = (ref_feet[:,2] - pred_feet[:,2]).unsqueeze(0)
                        #for i,(trans_sequence,sk_clip) in enumerate(zip(pred_seq,sk_clips)):
                        for i,trans_sequence in enumerate(pred_seq):
                            #trans_sequence[:,:,2] += translation
                            #display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt",soft_errors=soft_v2v[start_frame:start_frame+len(trans_sequence)]) #v2v_errors=per_vertex_acc_errors[start_frame:start_frame+len(trans_sequence)])
                            display_utils.write_amass_sequence(out_dir,"",trans_sequence,faces,start_frame,transfer_color,"tgt")
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
            with open(os.path.join(root_dir,test_args.baseline.strip("/")+"_mean.csv"),"w") as f:
                csv.writer(f,delimiter=" ").writerows([["v2p:",np.mean(all_v2p_errors),"v2v:",np.mean(all_v2v_errors)]])
            with open(os.path.join(root_dir,test_args.baseline.strip("/")+"_v2p.csv"),"w") as f:
                csv.writer(f,delimiter=",").writerows(name_and_v2p_errors)
            with open(os.path.join(root_dir,test_args.baseline.strip("/")+"_v2v.csv"),"w") as f:
                csv.writer(f,delimiter=",").writerows(name_and_v2v_errors)

    def def_trans_bunny(self,optimizer,epochs,loss_args,training_args,test_args,data_args,method=""):
        self.pointnet_network.load_state_dict(torch.load(training_args.weight_path+'_pointnet'))
        self.jacobian_first_order.load_state_dict(torch.load(training_args.weight_path+'_tangent'))
        self.transformer_network.load_state_dict(torch.load(training_args.weight_path+'_transformer'))
        self.acceleration_func.load_state_dict(torch.load(training_args.weight_path+'_acceleration'))
        dataloader = DataLoader(self.seq_dataset,training_args.batch,shuffle=False,num_workers=1)
        test_dataloader = DataLoader(self.test_seq_dataset,training_args.batch,shuffle=False,num_workers=1)
        test_faces = None
        test_sequence = []
        for _,test_data in enumerate(test_dataloader):
            test_vertices,test_faces = test_data
            test_vertices = test_vertices.float().cuda()
            test_faces = test_faces.long().cuda()
            test_sequence.append(test_vertices)
        #_mesh = trimesh.Trimesh(vertices=keyframes[0].squeeze().cpu().detach().numpy(),faces=test_faces.squeeze().cpu().detach().numpy(),process=False)
        #_mesh.export("test_bunny.ply")
        rest_shape = test_sequence[0]
        start_shape = test_sequence[0]
        transfer_color = np.array([255,127,80],dtype=float)/255.0
        source_color = np.array([95,158,160],dtype=float)/255.0
        #root_dir = os.path.join(test_args.int_output_dir,test_args.baseline)
        test_sequence = torch.cat(test_sequence,0)
        fframe = 0 ; lframe= len(test_sequence)
        with torch.no_grad():
            logger = logging.getLogger("Eval ")
            for bix,data in enumerate(dataloader):
                root_orient,pose_body,pose_hand,trans,betas,_,dmpls,fps,gender,_,source_name = data
                root_orient,pose_body,pose_hand,trans,betas,dmpls,fps = convert_to_float_tensors(root_orient,pose_body,pose_hand,trans,betas,dmpls,fps)
                root_orient = root_orient[:,fframe:lframe,:]
                pose_body = pose_body[:,fframe:lframe,:]
                pose_hand = pose_hand[:,fframe:lframe,:]
                trans = trans[:,fframe:lframe,:]
                dmpls = dmpls[:,fframe:lframe,:]
                betas = betas[:,:,fframe:lframe,:]
                zero_betas = torch.zeros(betas.size()).float().cuda()
                source_pose = torch.cat([root_orient.squeeze(),pose_body.squeeze(),pose_hand.squeeze()],-1)
                source_sequence,faces = get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,data_args.max_frames)
                #source_sequence,test_sequence = align_feet(source_sequence,test_sequence_tensor)
                source_clips,source_clip_times,sequence_times,split_indices = get_atomic_clips(source_sequence,self.frame_length,fps)
                #original_times = original_times/original_times[-1]
                clip_poses,_,_ = split_by_indices(source_pose,sequence_times,split_indices)
                test_clips,_,_ = split_by_indices(test_sequence,sequence_times,split_indices)
                source_start_shape = source_clips[0][0].unsqueeze(0)

                #start_shape = source_sequence[0].unsqueeze(0)
                #_,test_poisson_solver = self.get_poisson_system(test_clips[0][0].unsqueeze(0),faces)
                start_frame = 0
                clip_ix = 0
                frame_ix = 0
                for i,(clip,test_clip,clip_pose,times) in enumerate(zip(source_clips,test_clips,clip_poses,source_clip_times)):
                    #out_dir = os.path.join(test_args.int_output_dir,source_name[0].strip("/")+"_mushroom",test_args.baseline)
                    out_dir = os.path.join(test_args.int_output_dir,test_args.baseline)
                    seq_name = str(start_frame) + "_" + str(start_frame+len(clip))
                    #display_utils.write_amass_sequence(out_dir,seq_name,clip,faces,start_frame,source_color,"src")
                    #trans_sequence = self.one_sequence(clip,faces,times,loss_args,train=False,test_shape=start_shape,spf=1.0/fps.item(),test_faces=test_faces)
                    start_shape = test_clip[0].float().unsqueeze(0)
                    last_shape = test_clip[-1].unsqueeze(0)
                    trans_sequence = self.one_sequence(clip,faces,times,rest_shape.squeeze(),loss_args,train=False,test_shape=start_shape,spf=1.0/fps.item(),last_shape=last_shape,pose=clip_pose,betas=betas[0,0,0,:],encoding=training_args.encoding,test_faces=test_faces.squeeze())
                    #_mesh = trimesh.Trimesh(vertices=trans_sequence[0].squeeze().cpu().detach().numpy(),faces=test_faces.squeeze().cpu().detach().numpy(),process=False)
                    #_mesh.export("test_bunny.ply")
                    #exit()
                    ###start_shape = trans_sequence[-1].float().unsqueeze(0)
                    frame_ix += len(clip)

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    print(out_dir,seq_name)
                    display_utils.write_amass_sequence(out_dir,seq_name,clip,faces,start_frame,source_color,"src")
                    display_utils.write_amass_sequence(out_dir,"",trans_sequence,test_faces,start_frame,transfer_color,"tgt")
                    start_frame += len(clip) 
                    clip_ix += 1
                    message = colored("Rendered {0:3d} of {1:3d}".format(clip_ix,len(source_clips)),'blue')
                    logger.info(message)
