from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from datasets import Amass
import misc
import numpy as np
import torch
from human_body_prior.body_model.lbs import batch_rodrigues
import torch.nn as nn
from timeit import default_timer as timer
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_euler_angles
from smal.smal_torch import get_smal_model

def get_sequence(poses,trans,betas,betas_limbs,offset,max_frames):
    #root_orient,pose_body,pose_hand,trans,betas,dmpls = misc.squeeze(root_orient,pose_body,pose_hand,trans,betas,dmpls)
    #betas = betas[0]
    poses,trans,betas,betas_limbs,offset = misc.squeeze(poses,trans,betas,betas_limbs,offset)
    #print(pose_body.size(),root_orient.size(),pose_hand.size(),trans.size(),dmpls.size(),betas.size())
    poses = poses[:max_frames,:]
    trans = trans[:max_frames,:]
    betas = betas[:max_frames,:]
    betas_limbs = betas_limbs[:max_frames,:]
    offset = offset[:max_frames,:]
    smal = get_smal_model('cuda')
    poses_mat = axis_angle_to_matrix(poses)
    sequence_vertices, _, _ = smal(beta=betas, 
                                betas_limbs=betas_limbs, 
                                pose=poses_mat, 
                                trans=trans, 
                                vert_off_compact=offset)

    faces = misc.convert_to_long_tensors(smal.faces)[0]
    return sequence_vertices,faces

def get_atomic_clips(seq,frame_length,njf=False):
    prev_frame = seq[0] 
    current_start_ix = 0
    frame_seps = []
    prev_dist_from_t0 = 0
    split_indices = [0]
    prev_split_ix = 0
    for i in range(1,seq.size()[0]):
        _verts = seq[i,:,:]
        first_frame = seq[current_start_ix,:,:]
        frame_separation = torch.mean(torch.sqrt(torch.sum((_verts - prev_frame)**2,dim=1)))
        frame_seps.append(frame_separation.unsqueeze(0))
        prev_frame = _verts 
        dist_from_t0 = torch.mean(torch.sqrt(torch.sum((_verts - first_frame)**2,dim=1)))
        if dist_from_t0 < prev_dist_from_t0 or (i-prev_split_ix)>=frame_length:
            current_start_ix = i
            split_indices.append(i)
            prev_split_ix = i
            prev_dist_from_t0 = 0
        else:
            prev_dist_from_t0 = dist_from_t0

    frame_seps = torch.cat(frame_seps)
    total_frame_sep = torch.sum(frame_seps)
    cumulated_sum = torch.cumsum(frame_seps,0)
    seq_time_frames = cumulated_sum #/total_frame_sep
    seq_time_frames = torch.cat([torch.zeros(1).float().cuda(),seq_time_frames],0)
    #fps = 24
    #seconds_per_frame = 1.0/fps
    #end_time = seconds_per_frame*seq.size()[0]
    #seq_time_frames = torch.from_numpy(np.linspace(0,len(seq_time_frames)*seconds_per_frame,len(seq_time_frames))).float().cuda()

    #print(seq_time_frames,len(seq_time_frames),end_time)
    #exit()
    #sequence_times = torch.from_numpy(np.arange(0,end_time,seconds_per_frame)).float().cuda()*3

    #sequence_times = torch.from_numpy(np.arange(0,seq.size()[0],1)).float().cuda()

    sequence_times = seq_time_frames #*0.0

    clips = [] ; clip_times = []
    for i in range(1,len(split_indices)):
        start = split_indices[i-1]
        end = split_indices[i]
        clip = seq[start:end]
        times = sequence_times[start:end]
        clips.append(clip)
        translated_time = times-times[0]
        if not njf:
            clip_times.append(translated_time) #translated_time[-1])
        else:
            clip_times.append(translated_time)
    #for i in range(len(clip_times)):
    #    torch.mean(clip_times[i][1:]-clip_times[i][:-1])
    return clips,clip_times,sequence_times,split_indices

def split_by_indices(seq,sequence_times,split_indices,njf=False):
    clips = [] ; clip_times = [] 
    for i in range(1,len(split_indices)):
        start = split_indices[i-1]
        end = split_indices[i]
        clip = seq[start:end]
        times = sequence_times[start:end]
        clips.append(clip)
        translated_time = times #-times[0]
        if not njf:
            clip_times.append(translated_time) #translated_time[-1])
        else:
            clip_times.append(translated_time)
    return clips,clip_times,sequence_times

def get_clips(seq,frame_length):
    return torch.split(seq,frame_length)

def get_time_for_clip(seq):
    prev_frame = None
    frame_seps = []
    for i in range(seq.size()[0]):
        _verts = seq[i,:,:]
        if prev_frame is not None:
            frame_separation = torch.mean(torch.sqrt(torch.sum((_verts - prev_frame)**2,dim=1)))
            frame_seps.append(frame_separation.unsqueeze(0))

        prev_frame = _verts 
    frame_seps = torch.cat(frame_seps)
    total_frame_sep = torch.sum(frame_seps)
    cumulated_sum = torch.cumsum(frame_seps,0)
    seq_time_frames = cumulated_sum/total_frame_sep
    sequence_times = torch.cat([torch.from_numpy(np.array([0])).float().cuda(),seq_time_frames])

    return sequence_times

def get_time_for_jacs(jacs):
    prev_frame = None
    prev_frame_sep = None 
    frame_seps = []
    for i in range(jacs.size()[0]):
        _jac = jacs[i,:,:,:]
        if prev_frame is not None:
            frame_separation = torch.mean((_jac - prev_frame)**2)
            frame_seps.append(frame_separation.unsqueeze(0))

        prev_frame = _jac
    frame_seps = torch.cat(frame_seps)
    total_frame_sep = torch.sum(frame_seps)
    cumulated_sum = torch.cumsum(frame_seps,0)
    seq_time_frames = cumulated_sum/total_frame_sep
    sequence_times = torch.cat([torch.from_numpy(np.array([0])).float().cuda(),seq_time_frames])

    return sequence_times
