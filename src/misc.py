from asyncio.base_subprocess import BaseSubprocessTransport
import math
import numpy as np
import torch
from datetime import datetime,timezone
from pytorch3d.loss.point_mesh_distance import point_face_distance

def angle_between(vector_a, vector_b):
    #print(np.linalg.norm(vector_a),np.linalg.norm(vector_b))
    costheta = vector_a.dot(vector_b) / (np.linalg.norm(vector_a)*np.linalg.norm(vector_b))
    if math.isnan(costheta):
        print(costheta)
    return math.acos(costheta)

def cot(theta):
    return math.cos(theta) / math.sin(theta)

def tan(theta):
    return math.sin(theta) / math.cos(theta)

def compute_errors(pred_sequence,gt_sequence,faces):
    v2v_errors = torch.sqrt(torch.sum((pred_sequence - gt_sequence)**2,dim=-1))
    gt_triangles = gt_sequence[:,faces,:]
    num_shapes = pred_sequence.size()[0]
    num_verts = pred_sequence.size()[1]
    num_faces = faces.size()[0]
    pred_unwrapped = pred_sequence.view(-1,3)
    gt_triangles_unwrapped = gt_triangles.view(-1,3,3)
    shape_indices = torch.from_numpy(np.array([i*num_verts for i in range(num_shapes)])).long().cuda()
    triangle_indices = torch.from_numpy(np.array([i*num_faces for i in range(num_shapes)])).long().cuda()
    dists = point_face_distance(pred_unwrapped,shape_indices,gt_triangles_unwrapped,triangle_indices,num_verts)
    v2p_errors = dists.view(pred_sequence.size()[0],num_verts)

    return v2v_errors,v2p_errors

def soft_displacements(sequence,wodmpl_sequence):
    soft_disps = sequence - wodmpl_sequence
    return soft_disps

def compute_soft(pred_sequence,wodmpl_sequence):
    soft_errors = torch.sqrt(torch.sum((pred_sequence - wodmpl_sequence)**2,dim=-1))

    return soft_errors 

def shape_normalization_transforms(verts):
    bmin = np.min(verts,0)
    bmax = np.max(verts,0)
    diag = np.sqrt(np.sum((bmax-bmin)**2))
    bcenter = np.mean(verts,0)
    scale = 1.0/diag
    translation = -1*bcenter
    _verts = verts + translation

    '''
    _verts = _verts * scale


    bmin = np.min(_verts,0)
    bmax = np.max(_verts,0)
    diag = np.sqrt(np.sum((bmax-bmin)**2))
    '''

    return _verts,scale,translation

def align_feet(*align_list):
    ref_feet,_ = torch.min(align_list[0],dim=1,keepdim=True)

    aligned = [align_list[0]]
    for seq in align_list[1:]:
        align_feet,_ = torch.min(seq,dim=1,keepdim=True)
        translation = ref_feet[:,:,2] - align_feet[:,:,2]
        seq[:,:,2] += translation
        aligned.append(seq)

    return tuple(aligned)

def shape_normalization_transforms_pytorch(verts):

    scale = 1.0
    translation  = []
    if len(verts.size())==3:
        bcenters = torch.mean(verts,1,keepdim=True) #.repeat(1,verts.size()[1],1)
        translation = -1*bcenters
        _verts = verts + translation
        bcenters = torch.mean(_verts,1,keepdim=True) #.repeat(1,verts.size()[1],1)
    if len(verts.size())==2:
        '''
        bmin_t = torch.min(verts,0)
        bmax_t = torch.max(verts,0)
        bmin = bmin_t.values
        bmax = bmax_t.values
        diag = torch.sqrt(torch.sum((bmax-bmin)**2))
        '''
        bcenter = torch.mean(verts,0)

        '''
        scale = 1.0/diag
        '''

        translation = -1*bcenter
        _verts = verts + translation

        '''
        _verts = _verts * scale

        bmin_t = torch.min(_verts,0)
        bmax_t = torch.max(_verts,0)
        bmin = bmin_t.values
        bmax = bmax_t.values
        diag = torch.sqrt(torch.sum((bmax-bmin)**2))
        '''

    return _verts,scale,translation

def get_current_time():
    utc_dt = datetime.now(timezone.utc)
    dt = utc_dt.astimezone()
    hour = dt.hour ; minute = dt.minute ; second = dt.second

    return hour,minute,second

def convert_to_long_tensors_from_numpy(*args):
    converted = []
    for a in args:
        a_long = torch.from_numpy(a).long().cuda()
        converted.append(a_long)
    return tuple(converted)

def convert_to_float_tensors_from_numpy(*args):
    converted = []
    for a in args:
        a_float = torch.from_numpy(a).float().cuda()
        converted.append(a_float)
    return tuple(converted)


def convert_to_long_tensors(*args):
    converted = []
    for a in args:
        a_long = a.long().cuda()
        converted.append(a_long)
    return tuple(converted)

def convert_to_float_tensors(*args):
    converted = []
    for a in args:
        a_float = a.float().cuda()
        converted.append(a_float)
    return tuple(converted)

def squeeze(*args):
    converted = []
    for a in args:
        a_sq = a.squeeze()
        converted.append(a_sq)
    return tuple(converted)

