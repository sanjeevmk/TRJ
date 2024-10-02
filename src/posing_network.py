import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from fast_transformers.builders import TransformerEncoderBuilder,TransformerDecoderBuilder
from fast_transformers.transformers import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from fast_transformers.attention import AttentionLayer, FullAttention, LocalAttention
from torch.nn.functional import normalize
import math

class GetNormals(nn.Module):
    def __init__(self):
        super(GetNormals,self).__init__()

    def forward(self,triangles):
        #print(canonical_triangles)
        v2_v1 = triangles[:,2,:]-triangles[:,1,:]
        v2_v0 = triangles[:,2,:]-triangles[:,0,:]
        normal = torch.cross(v2_v0,v2_v1,dim=1) 
        z = torch.linalg.norm(normal,dim=1).unsqueeze(-1)
        normal = normal / z

        return normal

class JacFC(nn.Module):
    def __init__(self,n_dims,n_shapes):
        super(JacFC,self).__init__()
        self.fc1 = nn.Linear(n_shapes*n_dims,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,32)
        self.fc_tf = nn.Linear((n_dims-1),8)

    def forward(self,triangles):
        xt = F.relu(self.fc1(triangles))
        feature = self.fc2(xt)
        #feature = F.relu(self.fc3(xt2))

        return feature

class PoseFC(nn.Module):
    def __init__(self,n_dims,n_shapes):
        super(PoseFC,self).__init__()
        self.fc1 = nn.Linear(n_dims*n_shapes,32)
        self.fc2 = nn.Linear(32,32)

    def forward(self,triangles):
        xt = F.relu(self.fc1(triangles))
        feature = self.fc2(xt)
        #feature = F.relu(self.fc3(xt2))

        return feature

class CentroidFC(nn.Module):
    def __init__(self,n_dims,n_shapes):
        super(CentroidFC,self).__init__()
        self.fc1 = nn.Linear(n_dims*n_shapes,32)
        self.fc2 = nn.Linear(32,32)

    def forward(self,triangles):
        xt = F.relu(self.fc1(triangles))
        feature = self.fc2(xt)
        #feature = F.relu(self.fc3(xt2))

        return feature

class PosingNetwork(nn.Module):
    def __init__(self):
        super(PosingNetwork,self).__init__()
        seq_feat_size = 156 #+ 31*9
        tri_feat_size = 32 #32 
        jac_size = 32 
        betas_size = 0 

        self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,3*3)
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=512)
        self.model_dims = seq_feat_size + tri_feat_size + jac_size + 1
        self.time_ones = torch.ones(seq_feat_size+tri_feat_size+jac_size).float().cuda()
        self.jac_fc = JacFC(9,1)
        self.pose_fc = PoseFC(156,1) 
        self.centroid_fc = CentroidFC(6,1) 

    def forward(self,j0,cent_norm,pose_joints,betas,times):
        flat_j0 = j0.view(j0.size()[0],9).unsqueeze(1).repeat(1,pose_joints.size()[0],1)
        #cent_norm_feat = self.centroid_fc(cent_norm)
        cent_norm_time = cent_norm.unsqueeze(1).repeat(1,pose_joints.size()[0],1)

        betas_face_time = betas.unsqueeze(0).unsqueeze(0).repeat(j0.size()[0],pose_joints.size()[0],1)
        time_expanded = times.unsqueeze(0).unsqueeze(-1).repeat(j0.size()[0],1,1)
        #pose_feat = self.pose_fc(pose_joints)
        pose_joints_face = pose_joints.unsqueeze(0).repeat(j0.size()[0],1,1)

        j0_feat = self.jac_fc(flat_j0)
        xt = torch.cat([j0_feat,cent_norm_time,pose_joints_face],-1)
        #xt = torch.cat([j0_feat,cent_norm_time,pose_joints_face],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        xt = F.relu(self.fc3(xt))
        dj_dt = self.fc4(xt)
        J = flat_j0 + dj_dt
        J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)
        #J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)

        return J

class OursWKSPosingNetwork(nn.Module):
    def __init__(self):
        super(OursWKSPosingNetwork,self).__init__()
        seq_feat_size = 156 #+ 31*9
        #tri_feat_size = 32 + 2048 
        tri_feat_size = 32 + 100
        jac_size = 32 
        betas_size = 0 

        #self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size+1,512)
        #self.fc2 = nn.Linear(512,256)
        #self.fc3 = nn.Linear(256,128)
        #self.fc4 = nn.Linear(128,3*3)
        self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,3*3)
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=512)
        self.model_dims = seq_feat_size + tri_feat_size + jac_size + 1
        self.time_ones = torch.ones(seq_feat_size+tri_feat_size+jac_size).float().cuda()
        self.jac_fc = JacFC(9,1)
        self.pose_fc = PoseFC(156,1) 
        self.centroid_fc = CentroidFC(6,1) 
        self.d3f_project = CentroidFC(2048+6,1)

    def forward(self,j0,cent_norm,wks,pose_joints,betas,times):
        flat_j0 = j0.view(j0.size()[0],9).unsqueeze(1).repeat(1,pose_joints.size()[0],1)
        #cent_norm_feat = self.centroid_fc(cent_norm)
        #cent_norm_feat = self.d3f_project(torch.cat([d3f,cent_norm],-1))
        #cent_norm_d3f = torch.cat([d3f,cent_norm],-1)

        cent_norm_time = cent_norm.unsqueeze(1).expand(cent_norm.size()[0],pose_joints.size()[0],cent_norm.size()[-1])
        wks_time = wks.unsqueeze(1).expand(wks.size()[0],pose_joints.size()[0],wks.size()[-1])

        #betas_face_time = betas.unsqueeze(0).unsqueeze(0).expand(j0.size()[0],pose_joints.size()[0],betas.size()[0])
        time_expanded = times.unsqueeze(0).unsqueeze(-1).expand(j0.size()[0],times.size()[0],1)
        #pose_feat = self.pose_fc(pose_joints)
        pose_joints_face = pose_joints.unsqueeze(0).expand(j0.size()[0],pose_joints.size()[0],pose_joints.size()[-1])

        j0_feat = self.jac_fc(flat_j0)
        xt = torch.cat([j0_feat,cent_norm_time,wks_time,pose_joints_face],-1)
        #xt = torch.cat([j0_feat,cent_norm_time,pose_joints_face],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        xt = F.relu(self.fc3(xt))
        dj_dt = self.fc4(xt)
        J = flat_j0 + dj_dt
        J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)
        #J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)

        return J

class OursWKSPosingNetworkAnimals(nn.Module):
    def __init__(self):
        super(OursWKSPosingNetworkAnimals,self).__init__()
        seq_feat_size = 105 #+ 31*9
        #tri_feat_size = 32 + 2048 
        tri_feat_size = 32 + 100
        jac_size = 32 
        betas_size = 0 

        #self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size+1,512)
        #self.fc2 = nn.Linear(512,256)
        #self.fc3 = nn.Linear(256,128)
        #self.fc4 = nn.Linear(128,3*3)
        self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,3*3)
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=512)
        self.model_dims = seq_feat_size + tri_feat_size + jac_size + 1
        self.time_ones = torch.ones(seq_feat_size+tri_feat_size+jac_size).float().cuda()
        self.jac_fc = JacFC(9,1)
        self.pose_fc = PoseFC(156,1) 
        self.centroid_fc = CentroidFC(6,1) 
        self.d3f_project = CentroidFC(2048+6,1)

    def forward(self,j0,cent_norm,wks,pose_joints,betas,times):
        flat_j0 = j0.view(j0.size()[0],9).unsqueeze(1).repeat(1,pose_joints.size()[0],1)
        #cent_norm_feat = self.centroid_fc(cent_norm)
        #cent_norm_feat = self.d3f_project(torch.cat([d3f,cent_norm],-1))
        #cent_norm_d3f = torch.cat([d3f,cent_norm],-1)

        cent_norm_time = cent_norm.unsqueeze(1).expand(cent_norm.size()[0],pose_joints.size()[0],cent_norm.size()[-1])
        wks_time = wks.unsqueeze(1).expand(wks.size()[0],pose_joints.size()[0],wks.size()[-1])

        #betas_face_time = betas.unsqueeze(0).unsqueeze(0).expand(j0.size()[0],pose_joints.size()[0],betas.size()[0])
        time_expanded = times.unsqueeze(0).unsqueeze(-1).expand(j0.size()[0],times.size()[0],1)
        #pose_feat = self.pose_fc(pose_joints)
        pose_joints_face = pose_joints.unsqueeze(0).expand(j0.size()[0],pose_joints.size()[0],pose_joints.size()[-1])

        j0_feat = self.jac_fc(flat_j0)
        xt = torch.cat([j0_feat,cent_norm_time,wks_time,pose_joints_face],-1)
        #xt = torch.cat([j0_feat,cent_norm_time,pose_joints_face],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        xt = F.relu(self.fc3(xt))
        dj_dt = self.fc4(xt)
        J = flat_j0 + dj_dt
        J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)
        #J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)

        return J

class NJFWKSPosingNetwork(nn.Module):
    def __init__(self):
        super(NJFWKSPosingNetwork,self).__init__()
        seq_feat_size = 156 #+ 31*9
        #tri_feat_size = 32 + 2048 
        tri_feat_size = 32 + 100
        jac_size = 0 
        betas_size = 0 

        #self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size+1,512)
        #self.fc2 = nn.Linear(512,256)
        #self.fc3 = nn.Linear(256,128)
        #self.fc4 = nn.Linear(128,3*3)
        self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,3*3)
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=512)
        self.model_dims = seq_feat_size + tri_feat_size + jac_size + 1
        self.time_ones = torch.ones(seq_feat_size+tri_feat_size+jac_size).float().cuda()
        self.jac_fc = JacFC(9,1)
        self.pose_fc = PoseFC(156,1) 
        self.centroid_fc = CentroidFC(6,1) 
        self.d3f_project = CentroidFC(2048+6,1)

    def forward(self,j0,cent_norm,wks,pose_joints,betas,times):
        #cent_norm_feat = self.centroid_fc(cent_norm)
        #cent_norm_feat = self.d3f_project(torch.cat([d3f,cent_norm],-1))
        #cent_norm_d3f = torch.cat([d3f,cent_norm],-1)

        cent_norm_time = cent_norm.unsqueeze(1).expand(cent_norm.size()[0],pose_joints.size()[0],cent_norm.size()[-1])
        wks_time = wks.unsqueeze(1).expand(wks.size()[0],pose_joints.size()[0],wks.size()[-1])

        #betas_face_time = betas.unsqueeze(0).unsqueeze(0).expand(j0.size()[0],pose_joints.size()[0],betas.size()[0])
        time_expanded = times.unsqueeze(0).unsqueeze(-1).expand(j0.size()[0],times.size()[0],1)
        #pose_feat = self.pose_fc(pose_joints)
        pose_joints_face = pose_joints.unsqueeze(0).expand(j0.size()[0],pose_joints.size()[0],pose_joints.size()[-1])

        xt = torch.cat([cent_norm_time,wks_time,pose_joints_face],-1)
        #xt = torch.cat([j0_feat,cent_norm_time,pose_joints_face],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        xt = F.relu(self.fc3(xt))
        J = self.fc4(xt)
        J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)
        #J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)

        return J
    
class D4DPosingNetwork(nn.Module):
    def __init__(self):
        super(D4DPosingNetwork,self).__init__()
        seq_feat_size = 22 #+ 31*9
        #tri_feat_size = 32 + 2048 
        tri_feat_size = 32 + 100
        jac_size = 32 
        betas_size = 0 

        #self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size+1,512)
        #self.fc2 = nn.Linear(512,256)
        #self.fc3 = nn.Linear(256,128)
        #self.fc4 = nn.Linear(128,3*3)
        self.fc1 = nn.Linear(seq_feat_size + tri_feat_size + jac_size + betas_size+1,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,3*3)
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=512)
        self.model_dims = seq_feat_size + tri_feat_size + jac_size + 1
        self.time_ones = torch.ones(seq_feat_size+tri_feat_size+jac_size).float().cuda()
        self.jac_fc = JacFC(9,1)
        self.pose_fc = PoseFC(156,1) 
        self.centroid_fc = CentroidFC(6,1) 
        self.d3f_project = CentroidFC(2048+6,1)

    def forward(self,j0,cent_norm,wks,pose_joints,betas,times):
        flat_j0 = j0.view(j0.size()[0],9).unsqueeze(1).repeat(1,times.size()[0],1)
        #cent_norm_feat = self.centroid_fc(cent_norm)
        #cent_norm_feat = self.d3f_project(torch.cat([d3f,cent_norm],-1))
        #cent_norm_d3f = torch.cat([d3f,cent_norm],-1)

        cent_norm_time = cent_norm.unsqueeze(1).expand(cent_norm.size()[0],times.size()[0],cent_norm.size()[-1])
        wks_time = wks.unsqueeze(1).expand(wks.size()[0],times.size()[0],wks.size()[-1])
        #betas_face_time = betas.unsqueeze(0).unsqueeze(0).expand(j0.size()[0],pose_joints.size()[0],betas.size()[0])
        time_expanded = times.unsqueeze(0).unsqueeze(-1).expand(j0.size()[0],times.size()[0],1)
        #pose_feat = self.pose_fc(pose_joints)
        pose_joints_face = pose_joints.unsqueeze(0).expand(j0.size()[0],times.size()[0],pose_joints.size()[-1])
        print(pose_joints_face.size(),wks_time.size())
        exit()
        j0_feat = self.jac_fc(flat_j0)
        xt = torch.cat([j0_feat,cent_norm_time,wks_time,time_expanded],-1)
        #xt = torch.cat([j0_feat,cent_norm_time,pose_joints_face],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        xt = F.relu(self.fc3(xt))
        dj_dt = self.fc4(xt)
        J = flat_j0 + dj_dt
        J = J.view(j0.size()[0],times.size()[0],3,3).permute(1,0,2,3)
        #J = J.view(j0.size()[0],pose_joints.size()[0],3,3).permute(1,0,2,3)

        return J
