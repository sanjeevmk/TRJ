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

class CustomPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).float().cuda()
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        pe = torch.zeros(x.size()[0],1,x.size()[-1]).cuda()
        pe[:, 0, 0::2] = torch.sin(x[:,0,0::2] * self.div_term[:x[:,0,0::2].size()[-1]])
        pe[:, 0, 1::2] = torch.cos(x[:,0,1::2] * self.div_term[:x[:,0,1::2].size()[-1]])
        return pe


class CentroidNormalFC(nn.Module):
    def __init__(self,channel=6,local_out_dim=16,out_dim=128):
        super(CentroidNormalFC, self).__init__()
        #self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 32, 1)
        #self.bn1 = torch.nn.GroupNorm(num_groups=4,num_channels=128)
        #self.bn2 = torch.nn.GroupNorm(num_groups=4,num_channels=128)
        #self.bn3 = torch.nn.GroupNorm(num_groups=4,num_channels=128)
        #self.bn4 = torch.nn.GroupNorm(num_groups=4,num_channels=128)
        self.fc1 = torch.nn.Linear(local_out_dim,128)
        self.fc2 = torch.nn.Linear(128,out_dim)
        self.out_dim = out_dim
        self.local_out_dim = local_out_dim
        self.positional_encoding = CustomPositionalEncoding(channel)

    def forward(self, triangles):
        #triangles = self.positional_encoding(triangles.unsqueeze(1)).squeeze()
        x = triangles.unsqueeze(0)
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feature = self.conv4(x).permute(0,2,1).squeeze().contiguous()

        return feature

class VertexFC(nn.Module):
    def __init__(self,channel=3,local_out_dim=16,out_dim=128):
        super(VertexFC, self).__init__()
        #self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 32, 1)
        self.bn1 = torch.nn.GroupNorm(num_groups=4,num_channels=128)
        self.bn2 = torch.nn.GroupNorm(num_groups=4,num_channels=128)
        self.bn3 = torch.nn.GroupNorm(num_groups=4,num_channels=128)
        self.bn4 = torch.nn.GroupNorm(num_groups=4,num_channels=128)
        self.fc1 = torch.nn.Linear(local_out_dim,128)
        self.fc2 = torch.nn.Linear(128,out_dim)
        self.out_dim = out_dim
        self.local_out_dim = local_out_dim
        self.positional_encoding = CustomPositionalEncoding(channel)

    def forward(self, triangles):
        #triangles = self.positional_encoding(triangles.unsqueeze(1)).squeeze()
        x = triangles.unsqueeze(0)
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feature = self.conv4(x).permute(0,2,1).squeeze().contiguous()

        return feature

class SeqTransformerTimeJoints(nn.Module):
    def __init__(self,code_size):
        super(SeqTransformerTimeJoints,self).__init__()
        model_dimensions = 156 #+ 1 #+ 32
        model_dec_dimensions = 128 + 9 + 9 #  9+ 9 + 9
        d_keys = 32 
        d_values = 16 
        d_ff = 32 
        n_heads = 2
        self.num_parts = 50
        self.time_parts = 1
        self.transformer_encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(FullAttention(),model_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    model_dimensions,
                    activation="relu",
                    dropout=0.0,
                    d_ff=d_ff
                )
            ]
        )

        self.transformer_decoder = TransformerDecoder(
            [
                TransformerDecoderLayer(
                    AttentionLayer(FullAttention(),model_dec_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    AttentionLayer(FullAttention(),model_dec_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    model_dec_dimensions,
                    activation="relu",
                    dropout=0.0,
                    d_ff=d_ff
                )
            ]
        )


        self.positional_encoding = CustomPositionalEncoding(model_dimensions)
        self.minus_one = torch.ones(self.time_parts).float().cuda()
        self.batch = 32
        self.enc_l1 = nn.Linear(model_dimensions,64)
        self.dec_l1 = nn.Linear(model_dec_dimensions,32)
        self.dec_l2 = nn.Linear(32,9)

        self.l1 = nn.Linear(model_dimensions,32)
        self.l2 = nn.Linear(128,code_size) # Per Triangle
        self.l1_time = nn.Linear(model_dimensions,128)
        self.l2_time = nn.Linear(128,128) # Per Triangle
        self.mu = nn.Linear(128,128) # Per Triangle
        self.logvar = nn.Linear(128,128) # Per Triangle
        self.noise = torch.empty(128).float().cuda()
        self.positional_encoding = CustomPositionalEncoding(model_dimensions)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)

        return mu + eps * std

    def encoder(self,embeddings,times,sample=True):
        embeddings = embeddings.unsqueeze(0)
        expanded_times = times.unsqueeze(-1).unsqueeze(-1).repeat(1,1,embeddings.size()[-1])
        time_encodings = self.positional_encoding(expanded_times).permute(1,0,2)
        trans_out = self.transformer_encoder(embeddings+time_encodings)
        full_global_code = trans_out[:,0,:]
        #full_global_code = full_global_code.view(1,-1)
        full_global_code = self.l1(full_global_code.squeeze())
        #full_global_code = self.l2(full_global_code)

        return full_global_code
    
class SeqTransformerJacobians(nn.Module):
    def __init__(self,code_size):
        super(SeqTransformerJacobians,self).__init__()
        model_dimensions = 9 #+ 1 #+ 32
        model_dec_dimensions = 128 + 9 + 9 #  9+ 9 + 9
        d_keys = 32 
        d_values = 16 
        d_ff = 32 
        n_heads = 2
        self.num_parts = 50
        self.time_parts = 1
        self.transformer_encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(FullAttention(),model_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    model_dimensions,
                    activation="relu",
                    dropout=0.0,
                    d_ff=d_ff
                )
            ]
            #norm_layer=torch.nn.LayerNorm(model_dimensions)
        )

        self.transformer_decoder = TransformerDecoder(
            [
                TransformerDecoderLayer(
                    AttentionLayer(FullAttention(),model_dec_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    AttentionLayer(FullAttention(),model_dec_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    model_dec_dimensions,
                    activation="relu",
                    dropout=0.0,
                    d_ff=d_ff
                )
            ]
            #norm_layer=torch.nn.LayerNorm(model_dec_dimensions)
        )


        self.positional_encoding = CustomPositionalEncoding(model_dimensions)
        self.minus_one = torch.ones(self.time_parts).float().cuda()
        self.batch = 32
        self.enc_l1 = nn.Linear(model_dimensions,64)
        self.dec_l1 = nn.Linear(model_dec_dimensions,32)
        self.dec_l2 = nn.Linear(32,9)

        self.l1 = nn.Linear(model_dimensions,128)
        self.l2 = nn.Linear(128,code_size) # Per Triangle
        self.l1_time = nn.Linear(model_dimensions,128)
        self.l2_time = nn.Linear(128,128) # Per Triangle
        self.mu = nn.Linear(128,128) # Per Triangle
        self.logvar = nn.Linear(128,128) # Per Triangle
        self.noise = torch.empty(128).float().cuda()

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)

        return mu + eps * std

    def forward(self,embeddings,times,sample=True):
        #expanded_times = times.unsqueeze(0).unsqueeze(-1).repeat(embeddings.size()[0],1,embeddings.size()[-1])
        #time_encodings = self.positional_encoding(expanded_times)
        expanded_times = times.unsqueeze(-1).unsqueeze(-1).repeat(1,1,embeddings.size()[-1])
        time_encodings = self.positional_encoding(expanded_times).permute(1,0,2)
        #face_encodings = self.index_encoding(expanded_face_indices)
        trans_out = self.transformer_encoder(embeddings+time_encodings)
        attention_jacobians = trans_out[:,0,:].unsqueeze(1)

        return attention_jacobians
    
class SeqTransformerPoints(nn.Module):
    def __init__(self,code_size):
        super(SeqTransformerPoints,self).__init__()
        model_dimensions = 3 #+ 1 #+ 32
        model_dec_dimensions = 128 + 9 + 9 #  9+ 9 + 9
        d_keys = 32 
        d_values = 16 
        d_ff = 32 
        n_heads = 2
        self.num_parts = 50
        self.time_parts = 1
        self.transformer_encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(FullAttention(),model_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    model_dimensions,
                    activation="relu",
                    dropout=0.0,
                    d_ff=d_ff
                )
            ]
            #norm_layer=torch.nn.LayerNorm(model_dimensions)
        )

        self.transformer_decoder = TransformerDecoder(
            [
                TransformerDecoderLayer(
                    AttentionLayer(FullAttention(),model_dec_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    AttentionLayer(FullAttention(),model_dec_dimensions, n_heads,d_keys=d_keys,d_values=d_keys),
                    model_dec_dimensions,
                    activation="relu",
                    dropout=0.0,
                    d_ff=d_ff
                )
            ]
            #norm_layer=torch.nn.LayerNorm(model_dec_dimensions)
        )


        self.positional_encoding = CustomPositionalEncoding(model_dimensions)
        self.minus_one = torch.ones(self.time_parts).float().cuda()
        self.batch = 32
        self.enc_l1 = nn.Linear(model_dimensions,64)
        self.dec_l1 = nn.Linear(model_dec_dimensions,32)
        self.dec_l2 = nn.Linear(32,9)

        self.l1 = nn.Linear(model_dimensions,128)
        self.l2 = nn.Linear(128,code_size) # Per Triangle
        self.l1_time = nn.Linear(model_dimensions,128)
        self.l2_time = nn.Linear(128,128) # Per Triangle
        self.mu = nn.Linear(128,128) # Per Triangle
        self.logvar = nn.Linear(128,128) # Per Triangle
        self.noise = torch.empty(128).float().cuda()

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)

        return mu + eps * std

    def forward(self,embeddings,times,sample=True):
        #expanded_times = times.unsqueeze(0).unsqueeze(-1).repeat(embeddings.size()[0],1,embeddings.size()[-1])
        #time_encodings = self.positional_encoding(expanded_times)
        expanded_times = times.unsqueeze(-1).unsqueeze(-1).repeat(1,1,embeddings.size()[-1])
        time_encodings = self.positional_encoding(expanded_times).permute(1,0,2)
        #face_encodings = self.index_encoding(expanded_face_indices)
        trans_out = self.transformer_encoder(embeddings+time_encodings)
        attention_shapes = trans_out[:,0,:] #.unsqueeze(1)
        return attention_shapes
    
class BoneAngles(nn.Module):
    def __init__(self):
        super(BoneAngles,self).__init__()

    def forward(self,triangles,bone_pairs):
        centroids = torch.mean(triangles,dim=2)
        bone_angles = []
        timebatch_bone_pairs = centroids[:,bone_pairs,:]
        timebatch_bone_vectors_0 = timebatch_bone_pairs[:,:,0,:] - timebatch_bone_pairs[:,:,1,:]
        timebatch_bone_vectors_1 = timebatch_bone_pairs[:,:,3,:] - timebatch_bone_pairs[:,:,2,:]
        timebatch_bone_vectors_0 = timebatch_bone_vectors_0/torch.norm(timebatch_bone_vectors_0,dim=2,keepdim=True)
        timebatch_bone_vectors_1 = timebatch_bone_vectors_1/torch.norm(timebatch_bone_vectors_1,dim=2,keepdim=True)
        cos_ij = torch.einsum('bij,bij->bi', timebatch_bone_vectors_0, timebatch_bone_vectors_1)
        angles = torch.acos(cos_ij)
        return angles