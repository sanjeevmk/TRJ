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

def get_pose_of_shape(shape,root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,frame_ix):
    root_orient,pose_body,pose_hand,trans,dmpls = misc.squeeze(root_orient,pose_body,pose_hand,trans,dmpls)
    #print(pose_body.size(),root_orient.size(),pose_hand.size(),trans.size(),dmpls.size(),betas.size())
    root_orient = root_orient[frame_ix:frame_ix+1,:]
    root_mat = batch_rodrigues(root_orient).unsqueeze(0)
    pose_body = pose_body[frame_ix:frame_ix+1,:]
    pose_hand = pose_hand[frame_ix:frame_ix+1,:]
    trans = trans[frame_ix:frame_ix+1,:]
    dmpls = dmpls[frame_ix:frame_ix+1,:]
    betas = betas[:,:,frame_ix:frame_ix+1,:]
    betas = betas[0]
    full_pose = torch.cat([pose_body,pose_hand],-1)
    actual_mat = batch_rodrigues(full_pose.view(-1,3)).view([1,-1,3,3])
    if gender == 0:
        bm = Amass.bm_male
    elif gender == 1:
        bm = Amass.bm_female
    else:
        bm = Amass.bm_neutral
    pose_root = root_orient.clone().detach().float().cuda()
    trans = trans.clone().detach().float().cuda()
    pose_body_0 = torch.zeros(pose_body.size()).float().cuda()
    pose_hand_0 = torch.zeros(pose_hand.size()).float().cuda()
    full_pose = torch.nn.Parameter(torch.cat([pose_root,pose_body_0,pose_hand_0],-1))
    trans = torch.nn.Parameter(trans)
    optimizer = torch.optim.Adam([full_pose,trans],lr=1e-1)
    #pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
    #                            posedirs).view(batch_size, -1, 3)
    start = timer()
    mse = nn.MSELoss()
    bestshape = None
    for step in range(1000):
        optimizer.zero_grad()
        bm.zero_grad()
        body_params = {
                #'root_orient': pose_root,
                #'pose_body': pose_body_0,
                #'pose_hand': pose_hand_0,
                "return_dict":True,
                "full_pose" : full_pose,
                'trans': trans,
                'betas': betas[0].clone().detach(),
                'dmpls': dmpls.clone().detach()
        }

        body_data = bm(**{k:v for k,v in body_params.items()})
        loss = mse(body_data["v"],shape)
        loss.backward(retain_graph=True)
        optimizer.step()
        bestshape = body_data["v"]
        print(step,loss.item())
    print("time:",timer()-start)
    return bestshape.clone().detach()
    '''
    #true_body,_,_ = misc.shape_normalization_transforms_pytorch(body_data.v)
    ident = torch.eye(3).float().cuda()
    #pose_feature = torch.matmul(body_data.v.view(1,-1),torch.linalg.pinv(bm.posedirs)).view(1,-1,3,3) #.transpose(2,3)
    sol = torch.linalg.lstsq(bm.posedirs.transpose(0,1),body_data.v.view(1,-1).transpose(0,1)).solution
    pose_feature = sol.transpose(0,1).view(1,-1,3,3)
    rot_mat = pose_feature + ident
    #euler_angles = matrix_to_euler_angles(rot_mat,"XYZ")
    #_pose_body = euler_angles[:,:21,:].view(1,-1)
    #_pose_hand = euler_angles[:,21:,:].view(1,-1)
    pose = torch.cat([root_mat,rot_mat],1)
    #print(actual_mat[0][1])
    print("our actual")
    print(actual_mat[0][0])
    print()
    print(rot_mat[0][0])
    exit()
    body_params = {
            'root_orient': root_orient,
            #'pose_body': pose_body,
            #'pose_hand': pose_hand,
            'full_pose': pose,
            'trans': trans,
            'betas': betas[0],
            'dmpls': dmpls
    }

    body_data = bm.reverse(**{k:v for k,v in body_params.items()})

    body,_,_ = misc.shape_normalization_transforms_pytorch(body_data.v)

    return body 
    '''

def get_sequence(root_orient,pose_body,pose_hand,trans,betas,dmpls,gender,max_frames,use_dmpl=True):
    #root_orient,pose_body,pose_hand,trans,betas,dmpls = misc.squeeze(root_orient,pose_body,pose_hand,trans,betas,dmpls)
    #betas = betas[0]
    root_orient,pose_body,pose_hand,trans,dmpls = misc.squeeze(root_orient,pose_body,pose_hand,trans,dmpls)
    #print(pose_body.size(),root_orient.size(),pose_hand.size(),trans.size(),dmpls.size(),betas.size())
    root_orient = root_orient[:max_frames,:]
    pose_body = pose_body[:max_frames,:]
    pose_hand = pose_hand[:max_frames,:]
    trans = trans[:max_frames,:]
    dmpls = dmpls[:max_frames,:]
    betas = betas[:,:,:max_frames,:]
    betas = betas[0]
    if gender == 0:
        bm = Amass.bm_male
    elif gender == 1:
        bm = Amass.bm_female
    else:
        bm = Amass.bm_neutral
    bm.use_dmpl = use_dmpl

    seq_per_body_shape = []
    #for sh_ix in range(betas.size()[0]):
    for sh_ix in range(1):
        if use_dmpl:
            body_params = {
                    'root_orient': root_orient,
                    'pose_body': pose_body,
                    'pose_hand': pose_hand,
                    'trans': trans,
                    'betas': betas[sh_ix],
                    'dmpls': dmpls
            }
        else:
            body_params = {
                    'root_orient': root_orient,
                    'pose_body': pose_body,
                    'pose_hand': pose_hand,
                    'trans': trans,
                    'betas': betas[sh_ix]
            }

        body_data = bm(**{k:v for k,v in body_params.items()})

        body_shape_seq,_,_ = misc.shape_normalization_transforms_pytorch(body_data.v)

        seq_per_body_shape.append(body_shape_seq)

    faces = misc.convert_to_long_tensors(bm.f)[0]
    return seq_per_body_shape[0],faces

def get_atomic_clips_by_joints(seq,seq_pose,frame_length,fps,njf=False):
    seq_pose_matrices = axis_angle_to_matrix(seq_pose.view(seq_pose.size()[0],-1,3))
    seq_pose = matrix_to_euler_angles(seq_pose_matrices,convention='XYZ').view(seq_pose.size()[0],-1)

    prev_frame = seq_pose[0] 
    current_start_ix = 0
    frame_seps = []
    prev_dist_from_t0 = 0
    split_indices = [0]
    prev_split_ix = 0
    for i in range(1,seq_pose.size()[0]):
        _joints = seq_pose[i,:]
        first_frame = seq_pose[current_start_ix,:]
        frame_separation = torch.mean(torch.sqrt(torch.sum((_joints - prev_frame)**2,dim=0)))
        frame_seps.append(frame_separation.unsqueeze(0))
        prev_frame = _joints
        dist_from_t0 = torch.mean(torch.sqrt(torch.sum((_joints - first_frame)**2,dim=0)))
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

    seconds_per_frame = 1.0/fps.item()
    end_time = seconds_per_frame*seq.size()[0]
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

def get_atomic_clips(seq,frame_length,fps,njf=False,fixed_time=False):
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

    if fixed_time:
        fps = 30
        seconds_per_frame = 1.0/fps
        seq_time_frames = torch.from_numpy(np.linspace(0,len(seq_time_frames)*seconds_per_frame,len(seq_time_frames))).float().cuda()

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
