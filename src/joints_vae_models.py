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
    