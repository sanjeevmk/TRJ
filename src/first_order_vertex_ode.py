import torch
from torch import nn

class VertexODE(nn.Module):
    def __init__(self,vert_first_order_func):
        super(VertexODE, self).__init__()
        self.vert_first_order_func = vert_first_order_func 

        #self.prev_t = 0.0
        #self.time_index = 0
        #self.code_size = 16 
        #self.per_time_code_size = 9 
        self.num_betas = 16
        self.zeros_num_verts = torch.zeros(1,1).float().cuda()
        self.zeros_betas = torch.zeros(1,16).float().cuda()

    def forward(self,t,state):
        current_state = state[0][0]
        start_index = 0
        end_index = start_index+1
        num_vertices = current_state[start_index:end_index].int()

        start_index = end_index
        end_index = start_index + num_vertices*3
        v0 = current_state[start_index:end_index].view(num_vertices,3)

        start_index = end_index
        end_index = start_index + num_vertices*3
        v0 = current_state[start_index:end_index].view(num_vertices,3)

        start_index = end_index
        end_index = start_index + num_vertices*32
        v0_feat = current_state[start_index:end_index].view(num_vertices,32)

        start_index = end_index
        end_index = start_index + num_vertices*100
        d3f = current_state[start_index:end_index].view(num_vertices,100).long()

        start_index = end_index
        end_index = start_index + self.num_betas
        betas = current_state[start_index:end_index].view(1,self.num_betas)

        dv_dt = self.vert_first_order_func(v0,v0_feat,d3f,betas,t)

        zeros_vertices = torch.zeros(1,num_vertices*3).float().cuda()
        zeros_vertex_features = torch.zeros(1,num_vertices*32).float().cuda()
        zeros_wks_features = torch.zeros(1,num_vertices*100).float().cuda()
        dstate_dt = tuple([torch.cat([self.zeros_num_verts,dv_dt.view(1,-1),zeros_vertices,zeros_vertex_features,zeros_wks_features,self.zeros_betas],1)])
        return dstate_dt
