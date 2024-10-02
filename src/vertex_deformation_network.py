import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fast_transformers.builders import TransformerEncoderBuilder,TransformerDecoderBuilder
from fast_transformers.transformers import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from fast_transformers.attention import AttentionLayer, FullAttention, LocalAttention
from torch.nn.functional import normalize
import math
from util_networks import CustomPositionalEncoding
from torch_geometric.nn import ChebConv as GCN 

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

class MeshConv(nn.Module):
    def __init__(self,n_dims,n_shapes):
        super(MeshConv,self).__init__()
        self.l1 = GCN(n_shapes*n_dims,64,K=2)
        self.l2 = GCN(64,32,K=2)
        self.fc3 = nn.Linear(32,32)
        self.fc_tf = nn.Linear((n_dims-1),8)

    def forward(self,verts,edges):
        xt = F.relu(self.l1(verts,edges.t()))
        feature = self.l2(xt,edges.t())
        #feature = F.relu(self.fc3(xt2))

        return feature

class VertFC(nn.Module):
    def __init__(self,n_dims,n_shapes):
        super(VertFC,self).__init__()
        self.fc1 = nn.Linear(n_shapes*n_dims,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,32)
        self.fc_tf = nn.Linear((n_dims-1),8)

    def forward(self,verts):
        xt = F.relu(self.fc1(verts))
        feature = self.fc2(xt)
        #feature = F.relu(self.fc3(xt2))

        return feature

class BetaFC(nn.Module):
    def __init__(self,n_dims,n_shapes):
        super(BetaFC,self).__init__()
        self.fc1 = nn.Linear(n_shapes*n_dims,64)
        self.fc2 = nn.Linear(64,256)
        self.fc3 = nn.Linear(32,32)
        self.fc_tf = nn.Linear((n_dims-1),8)
        self.positional_encoding = CustomPositionalEncoding(16)

    def forward(self,betas):
        betas = betas.unsqueeze(0) #.permute(0,2,1)
        betas = self.positional_encoding(betas).squeeze().unsqueeze(0)
        xt = F.relu(self.fc1(betas[0]))
        feature = self.fc2(xt)
        #feature = F.relu(self.fc3(xt2))

        return feature

class VertexDeformationNetwork(nn.Module):
    def __init__(self):
        super(VertexDeformationNetwork,self).__init__()
        vert_feat_size = 32 
        betas_size = 256 
        v0_size = 3
        wks_size = 100
        time_size = 1

        self.fc1 = nn.Linear(v0_size + vert_feat_size + wks_size + betas_size + time_size,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,3)
        self.features = MeshConv(3,1)

        self.beta_fc = BetaFC(16,1)

    def forward(self,v0,v0_feat,wks,betas,t):
        #flat_primary_attention = primary_attention.view(v0.size()[0],9)
        #flat_previous_attention = previous_attention.view(v0.size()[0],9)
        betas_feat = self.beta_fc(betas)
        betas_verts = betas_feat.repeat(v0.size()[0],1)
        expanded_time = t.unsqueeze(0).repeat(v0.size()[0],1)
        xt = torch.cat([v0,v0_feat,wks,betas_verts,expanded_time],-1)
        xt = F.relu(self.fc1(xt))
        xt = F.relu(self.fc2(xt))
        dv_dt = self.fc3(xt)
        #xt = torch.cat([v0,primary_shapes_attention,previous_shapes_attention,betas_verts,expanded_time],-1)
        #xt = torch.cat([j0_feat,previous_attention_feat,cent_norm,betas_face,expanded_time],-1)

        return dv_dt
 
class VertexAccelerationField(nn.Module):
    def __init__(self):
        super(VertexAccelerationField,self).__init__()
        vertex_feature_size = 32 
        betas_size = 32 #256 
        self.noise_feat = 0 #16
        self.time_size = 1

        self.fc1 = nn.Linear(3*3 + betas_size + self.time_size,512)
        self.fc2 = nn.Linear(512,3)
        self.beta_feature = BetaFC(16,1)
        self.vertex_feature = VertFC(3,4)
        self.normal = GetNormals()
        self.zero = torch.from_numpy(np.array([0.0])).float().cuda()

    def forward(self,initial_velocity,initial_vertex,current_velocity,current_vertex,primary_attention,betas,t):
        expanded_time = t.unsqueeze(0).repeat(initial_vertex.size()[0],1)
        #vert_feat = self.vertex_feature(torch.cat([initial_velocity,current_velocity,initial_vertex,current_vertex],-1))
        beta_feature = self.beta_feature(betas)
        beta_feature = beta_feature.unsqueeze(0).repeat(initial_vertex.size()[0],1)
        xt = torch.cat([initial_velocity,initial_vertex,primary_attention,beta_feature,expanded_time],-1)

        xt = F.relu(self.fc1(xt))
        d2v_dt2 = self.fc2(xt)
        return d2v_dt2

class VertexVelocityIntegrator(nn.Module):
    def __init__(self):
        super(VertexVelocityIntegrator,self).__init__()

    def forward(self,current_velocity):
        dv_dt = current_velocity
        return dv_dt 
   
