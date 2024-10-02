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
from util_networks import CustomPositionalEncoding

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

class BetaFC(nn.Module):
    def __init__(self,n_dims,n_shapes):
        super(BetaFC,self).__init__()
        self.fc1 = nn.Linear(n_shapes*n_dims,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,32)
        self.fc_tf = nn.Linear((n_dims-1),8)
        self.positional_encoding = CustomPositionalEncoding(n_dims)

    def forward(self,betas):
        betas = betas.unsqueeze(0) #.permute(0,2,1)
        betas = self.positional_encoding(betas).squeeze().unsqueeze(0)
        xt = F.relu(self.fc1(betas[0]))
        feature = self.fc2(xt)
        #feature = F.relu(self.fc3(xt2))

        return feature

class JacobianNetwork(nn.Module):
    def __init__(self,aug_dim=0):
        super(JacobianNetwork,self).__init__()
        tri_feat_size = 0 #if cn passed to ode else 0
        jac_size = 32 
        betas_size = 32 
        previous_attention_size = 32 
        primary_attention_size = 32 #32
        pose_feat = 0 
        time_size = 1
        self.aug_dim = aug_dim

        #self.fc1 = nn.Linear(primary_attention_size + previous_attention_size + tri_feat_size + jac_size + betas_size + time_size,512)
        self.fc1 = nn.Linear(primary_attention_size + previous_attention_size + jac_size + pose_feat + tri_feat_size + betas_size + time_size,512)
        self.fc2 = nn.Linear(512,256)
        if self.aug_dim == 0:
            self.fc3 = nn.Linear(256,9)
        else:
            self.fc3 = nn.Linear(256,9+self.aug_dim)

        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=512)
        self.jac_fc = JacFC(9,1)
        self.primary_attention = JacFC(9,1)
        self.previous_attention = JacFC(9,1)
        self.beta_fc = BetaFC(16,1)

    def forward(self,j0,primary_attention,previous_attention,betas,cn_feat,pose_enc,target_j,t):
        flat_j0 = j0.view(j0.size()[0],9)
        flat_primary_attention = primary_attention.view(j0.size()[0],9)
        flat_previous_attention = previous_attention.view(j0.size()[0],9)
        betas_feat = self.beta_fc(betas)
        betas_face = betas_feat.unsqueeze(0).expand(j0.size()[0],betas_feat.size()[0])
        expanded_time = t.unsqueeze(0).expand(j0.size()[0],1)
        j0_feat = self.jac_fc(flat_j0)
        primary_attention_feat = self.jac_fc(flat_primary_attention)
        previous_attention_feat = self.jac_fc(flat_previous_attention)

        #if no CN passed to ode
        xt = torch.cat([j0_feat,primary_attention_feat,previous_attention_feat,betas_face,expanded_time],-1)

        #if CN passed to ode
        #xt = torch.cat([j0_feat,primary_attention_feat,previous_attention_feat,cn_feat,betas_face,expanded_time],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        #xt = self.fc2(xt)
        dj_dt = self.fc3(xt)

        return dj_dt
    
class JacobianNetworkAnimals(nn.Module):
    def __init__(self):
        super(JacobianNetworkAnimals,self).__init__()
        tri_feat_size = 0 #32
        jac_size = 32 
        betas_size = 256 
        previous_attention_size = 32 
        primary_attention_size = 32 #32
        pose_feat = 0 
        time_size = 1

        #self.fc1 = nn.Linear(primary_attention_size + previous_attention_size + tri_feat_size + jac_size + betas_size + time_size,512)
        self.fc1 = nn.Linear(primary_attention_size + previous_attention_size + jac_size + pose_feat + tri_feat_size + betas_size + time_size,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,9)
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=512)
        self.jac_fc = JacFC(9,1)
        self.primary_attention = JacFC(9,1)
        self.previous_attention = JacFC(9,1)
        self.beta_fc = BetaFC(37,1)

    def forward(self,j0,primary_attention,previous_attention,betas,target_j,t):
        flat_j0 = j0.view(j0.size()[0],9)
        flat_primary_attention = primary_attention.view(j0.size()[0],9)
        flat_previous_attention = previous_attention.view(j0.size()[0],9)
        betas_feat = self.beta_fc(betas)
        betas_face = betas_feat.unsqueeze(0).expand(j0.size()[0],betas_feat.size()[0])
        expanded_time = t.unsqueeze(0).expand(j0.size()[0],1)
        j0_feat = self.jac_fc(flat_j0)
        primary_attention_feat = self.jac_fc(flat_primary_attention)
        previous_attention_feat = self.jac_fc(flat_previous_attention)
        #xt = torch.cat([primary_attention_feat,previous_attention_feat,betas_face,expanded_time],-1)
        xt = torch.cat([j0_feat,primary_attention_feat,previous_attention_feat,betas_face,expanded_time],-1)
        #xt = torch.cat([j0_feat,previous_attention_feat,betas_face,expanded_time],-1)
        #xt = torch.cat([j0_feat,primary_attention_feat,betas_face,expanded_time],-1)
        #xt = torch.cat([j0_feat,previous_attention_feat,cent_norm,betas_face,expanded_time],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        dj_dt = self.fc3(xt)

        return dj_dt
    
class JacobianNetworkD4D(nn.Module):
    def __init__(self):
        super(JacobianNetworkD4D,self).__init__()
        tri_feat_size = 0 #32
        jac_size = 32 
        #betas_size = 0 #256 
        betas_size = 256 #256 
        previous_attention_size = 32 
        primary_attention_size = 32 #32
        pose_feat = 0 
        time_size = 1

        #self.fc1 = nn.Linear(primary_attention_size + previous_attention_size + tri_feat_size + jac_size + betas_size + time_size,512)
        self.fc1 = nn.Linear(primary_attention_size + previous_attention_size + jac_size + pose_feat + tri_feat_size + betas_size + time_size,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,9)
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=512)
        self.jac_fc = JacFC(9,1)
        self.primary_attention = JacFC(9,1)
        self.previous_attention = JacFC(9,1)
        self.beta_fc = BetaFC(16,1)

    def forward(self,j0,primary_attention,previous_attention,betas,target_j,t):
        flat_j0 = j0.view(j0.size()[0],9)
        flat_primary_attention = primary_attention.view(j0.size()[0],9)
        flat_previous_attention = previous_attention.view(j0.size()[0],9)
        betas_feat = self.beta_fc(betas)
        betas_face = betas_feat.unsqueeze(0).expand(j0.size()[0],betas_feat.size()[0])
        expanded_time = t.unsqueeze(0).expand(j0.size()[0],1)
        j0_feat = self.jac_fc(flat_j0)
        primary_attention_feat = self.jac_fc(flat_primary_attention)
        previous_attention_feat = self.jac_fc(flat_previous_attention)
        #xt = torch.cat([primary_attention_feat,previous_attention_feat,betas_face,expanded_time],-1)
        xt = torch.cat([j0_feat,primary_attention_feat,previous_attention_feat,betas_face,expanded_time],-1)
        #xt = torch.cat([j0_feat,primary_attention_feat,previous_attention_feat,expanded_time],-1)
        #xt = torch.cat([j0_feat,previous_attention_feat,betas_face,expanded_time],-1)
        #xt = torch.cat([j0_feat,primary_attention_feat,betas_face,expanded_time],-1)
        #xt = torch.cat([j0_feat,previous_attention_feat,cent_norm,betas_face,expanded_time],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        dj_dt = self.fc3(xt)

        return dj_dt
    