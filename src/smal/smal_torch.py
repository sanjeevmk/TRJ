"""
PyTorch implementation of the SMAL/SMPL model
see:
    1.) https://github.com/silviazuffi/smalst/blob/master/smal_model/smal_torch.py
    2.) https://github.com/benjiebob/SMALify/blob/master/smal_model/smal_torch.py
"""
import numpy as np
import torch
from torch import nn
#----------------------------------------------
from smal.batch_lbs import batch_rodrigues, batch_global_rigid_transformation_biggs, get_beta_scale_mask

CONFIG_PATH = 'custom_smal_config2.pt'

CANONICAL_COLORS = np.array([
    [158, 59, 98], # upper_right [paw, middle, top]
    [151, 56, 92],
    [135, 47, 82],
    [149, 146, 105], # lower_right [paw, middle, top]
    [118, 99, 77],
    [140, 103, 90],
    [51, 60, 96], # upper_left [paw, middle, top]
    [47, 59, 95],
    [45, 50, 82],
    [48, 145, 104], # lower_left [paw, middle, top]
    [50, 104, 80],
    [55, 102, 86],
    [69, 97, 67], # tail [start, end]
    [67, 121, 69],
    [146, 24, 85],
    [14, 20, 73], # ear base [left, right]
    [100, 1, 99], # nose, chin
    [89, 4, 87],
    [154, 22, 84],
    [10, 21, 72], # ear tip [left, right]
    [125, 16, 99],
    [77, 18, 99], # eyes [left, right]
    [72, 4, 84], # withers, throat
    [91, 38, 91],
    [0, 0,  0]
]).reshape((25,3))


CANONICAL_MODEL_JOINTS_REFINED = [
  41, 9, 8, # upper_left [paw, middle, top]
  43, 19, 18, # lower_left [paw, middle, top]
  42, 13, 12, # upper_right [paw, middle, top]
  44, 23, 22, # lower_right [paw, middle, top]
  25, 31, # tail [start, end]
  33, 34, # ear base [left, right]
  35, 36, # nose, chin
  38, 37, # ear tip [left, right]
  39, 40, # eyes [left, right]
  46, 45, # withers, throat
  28] # tail middle

class SMAL(nn.Module):        
    def __init__(self, buffer_path, mode_original=False, scale_factor=1.):
        super(SMAL, self).__init__()

        self.mode_original = mode_original
        buffer = torch.load(buffer_path)

        self.logscale_part_list = ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
        
        #TORCHFLOAT32 (7, 105)
        self.betas_scale_mask = get_beta_scale_mask(part_list=self.logscale_part_list)

        #FACES TORCHINT64 (7774, 3)
        self.register_buffer('faces', buffer['faces'])

        #Scaling mesh option ------
        if scale_factor == 1:
            # Mean template vertices (TORCHFLOAT32 (3889, 3))
            self.register_buffer('v_template', buffer['v_template'])
        else:
            v_template_scale = scale_mesh_from_normal(buffer['v_template'].unsqueeze(0), self.faces.unsqueeze(0), scale=scale_factor)
            self.register_buffer('v_template', v_template_scale.squeeze())
        
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]

        # Shape blend shape basis (TORCHFlOAT32 (78, 11667))
        self.register_buffer('shapedirs', buffer['shapedirs'])

        # Regressor for joint locations given shape (TORCHFLOAT32 (3889,35))
        self.register_buffer('J_regressor', buffer['J_regressor'])

        # Pose blend shape basis (TORCHFLOAT32 (306, 11667))
        self.register_buffer('posedirs', buffer['posedirs'])
        
        # indices of parents for each joints ((NPINT32, (35,)))
        self.parents = buffer['kintree_table'].numpy()

        # LBS weights (TORCHFLOAT32, (3889, 35))
        self.register_buffer('weights', buffer['weights'])

        # Symmetric deformation -----
        self.register_buffer('inds_back', buffer['inds_back'])
        self.n_center = buffer['n_center']
        self.n_left = buffer['n_left']
        self.sl = buffer['s_left']


    def compute_symmetric_vert_off(self, vert_off_compact:torch.Tensor, device:str) -> torch.Tensor:
        '''
        Args:
            - vert_off_compact (BATCH,5901) TORCHFLOAT32 [DEVICE]
        Returns:
            - vertex_offsets (BATCH, 3889, 3) TORCHFLOAT32 [DEVICE]
        '''

        # vert_off_compact (BATCH, 2*self.n_center + 3*self.n_left)
        zero_vec = torch.zeros((vert_off_compact.shape[0], self.n_center)).to(device)
        
        half_vertex_offsets_center = torch.stack((
            vert_off_compact[:, :self.n_center], \
            zero_vec, \
            vert_off_compact[:, self.n_center:2*self.n_center]), axis=1)
        
        half_vertex_offsets_left = torch.stack((
            vert_off_compact[:, self.sl:self.sl+self.n_left], \
            vert_off_compact[:, self.sl+self.n_left:self.sl+2*self.n_left], \
            vert_off_compact[:, self.sl+2*self.n_left:self.sl+3*self.n_left]), axis=1)
        
        half_vertex_offsets_right = torch.stack((
            vert_off_compact[:, self.sl:self.sl+self.n_left], \
            - vert_off_compact[:, self.sl+self.n_left:self.sl+2*self.n_left], \
            vert_off_compact[:, self.sl+2*self.n_left:self.sl+3*self.n_left]), axis=1)
        
        # (bs, 3, 3889)
        half_vertex_offsets_tot = torch.cat((
            half_vertex_offsets_center, 
            half_vertex_offsets_left, 
            half_vertex_offsets_right), dim=2)       

        vertex_offsets = torch.index_select(half_vertex_offsets_tot, dim=2, index=self.inds_back.to(half_vertex_offsets_tot.device)).permute((0, 2, 1))     # (bs, 3889, 3)

        return vertex_offsets


    def __call__(self, beta, betas_limbs, theta=None, pose=None, trans=None, vert_off_compact=None, return_transformation=False):

        device = beta.device
        nBetas = beta.shape[1]
        betas_logscale = betas_limbs

        if (theta is None) and (pose is None):
            raise ValueError("Either pose (rotation matrices NxNJointsx3x3) or theta (axis angle BSxNJointsx3) must be given")
        elif (theta is not None) and (pose is not None):
            raise ValueError("Not both pose (rotation matrices NxNJointsx3x3) and theta (axis angle BSxNJointsx3) can be given")

        # add possibility to have additional vertex offsets
        if vert_off_compact is None:
            vertex_offsets = torch.zeros_like(self.v_template)
        else:
            vertex_offsets = self.compute_symmetric_vert_off(vert_off_compact, device)

        # 1. Add shape blend shapes
        v_shape_blend_shape = torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        v_shaped = self.v_template + v_shape_blend_shape + vertex_offsets

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        if pose is None:
            Rs = torch.reshape(batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3])
        else:
            Rs = pose
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(device=device), [-1, 306])

        # 3. Add pose blend shapes
        v_pose_blend_shape = torch.reshape(torch.matmul(pose_feature, self.posedirs), [-1, self.size[0], self.size[1]])
        v_posed = v_pose_blend_shape + v_shaped
        
        #Add corrections of bone lengths to the template
        #[NBATCH, 105]
        betas_scale = torch.exp(betas_logscale @ self.betas_scale_mask.to(betas_logscale.device))
        #[NBATCH, 35, 3]
        scaling_factors = betas_scale.reshape(-1, 35, 3)
        scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)

        # 4. Get the global joint location (BATCH, 35, 3) + relative joint transformations for LBS (BATCH, 35, 4, 4)
        J_transformed, A = batch_global_rigid_transformation_biggs(Rs, J, self.parents, scale_factors_3x3)

        #Recenter the mesh (original SMAL centered at beginning of the tail)
        if not self.mode_original:
            kp1, kp2 = 37,1808
            v_center = (v_posed[:,kp1,None] + v_posed[:,kp2,None]) / 2
            v_posed = v_posed - v_center

        # 5. Do skinning ->(BATCH, 3889, 35):
        num_batch = Rs.shape[0]
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])
        
        #The weighted transformation to apply on v_posed  ->(BATCH, 3889, 4, 4)
        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])),
                [num_batch, -1, 4, 4])
        
        #-> (BATCH, 3889, 4)
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=device)], 2)
        
        #->(BATCH, 3889, 4)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        #->(BATCH, 3889, 3)
        verts = v_homo[:, :, :3, 0]

        # 6. Apply translation:
        if trans is None:
            trans = torch.zeros((num_batch,3)).to(device=device)
        v_trans =  trans[:,None,:]
        verts = verts + v_trans
        
        # 7. Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        joints = torch.cat([
            joints,
            verts[:, None, 1863],   # end_of_nose
            verts[:, None, 26],     # chin
            verts[:, None, 2124],   # right ear tip
            verts[:, None, 150],    # left ear tip
            verts[:, None, 3055],   # left eye
            verts[:, None, 1097],   # right eye
            # new: add paw keypoints, not joint locations -> bottom, rather in front
            # remark: when i look in the animals direction, left and right are exchanged
            verts[:, None, 1330],   # front paw, right
            verts[:, None, 3282],   # front paw, left
            verts[:, None, 1521],   # back paw, right
            verts[:, None, 3473],   # back paw, left
            verts[:, None, 6],      # throat
            verts[:, None, 20],     # withers
            ], dim = 1)

        #Return all tensors of the transformation
        if return_transformation:
            return (v_shape_blend_shape, v_pose_blend_shape, vertex_offsets, v_center, A, v_trans)
        else:
            return verts, joints[:, CANONICAL_MODEL_JOINTS_REFINED, :], Rs

    def get_lbo_weights(self):
        return self.weights.detach().clone()

    
def lbs_invert_points(X:torch.Tensor, A:torch.Tensor, Tl:torch.Tensor, X_lbs_weights:torch.Tensor, X_offset:torch.Tensor):
    '''
    Args:
        - X (BATCH, N_pts, 3) TORCHFLOAT32 [DEVICE]
        - A (BATCH, 35, 4, 4) TORCHFLOAT32 [DEVICE]
        - Tl (BATCH, 1, 3) TORCHFLOAT32 [DEVICE]
        - X_lbs_weights (BATCH, N_pts, 35) TORCHFLOAT32 [DEVICE]
        - X_offset (BATCH, N_pts, N_verts) TORCHFLOAT32 [DEVICE]
    Returns:
        - X_inverted (BATCH, N_pts, 3) TORCHFLOAT32 [DEVICE]
    '''
    BATCH, N_pts, _ = X.shape
    device = X.device
    
    #Inverse transformation A (BATCH, 35, 4, 4) -> (BATCH, N_pts, 4, 4)
    A_inv = torch.inverse(A)
    T_inv = torch.matmul(X_lbs_weights, A_inv.reshape(BATCH, 35, 16)).reshape(BATCH, N_pts, 4, 4)

    #-> (BATCH, N_pts, 4)
    X = X - Tl
    X_homo = torch.cat([X, torch.ones([BATCH, N_pts, 1]).to(device=device)], -1)
    
    #->(BATCH, N_pts, 3)
    X_homo = torch.matmul(T_inv, X_homo.unsqueeze(-1))[...,:3,0]

    #->(BATCH, N_pts, 3)
    X_inverted = X_homo - X_offset

    return X_inverted

def get_smal_model(device, mode_original=False, scale_factor=1.) -> SMAL:
    smal = SMAL(buffer_path=CONFIG_PATH, mode_original=mode_original, scale_factor=scale_factor).to(device)    
    return smal

def get_smal_original_barc_bite_model(device):
    return get_smal_model(device, mode_original=True)
