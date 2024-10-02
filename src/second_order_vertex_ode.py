import torch
from torch import nn

class VertexODE(nn.Module):
    def __init__(self,vert_accleration_func,vert_velocity_integrator):
        super(VertexODE, self).__init__()
        self.vert_acceleration_func = vert_accleration_func
        self.vert_velocity_integrator = vert_velocity_integrator

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
        end_index = start_index+num_vertices*3
        current_velocity = current_state[start_index:end_index].view(num_vertices,3)

        start_index = end_index
        end_index = start_index + num_vertices*3
        current_vertex = current_state[start_index:end_index].view(num_vertices,3)

        start_index = end_index
        end_index = start_index+num_vertices*3
        initial_velocity = current_state[start_index:end_index].view(num_vertices,3)

        start_index = end_index
        end_index = start_index + num_vertices*3
        initial_vertex = current_state[start_index:end_index].view(num_vertices,3)

        start_index = end_index
        end_index = start_index + num_vertices*3
        primary_shapes_attention = current_state[start_index:end_index].view(num_vertices,3)

        start_index = end_index
        end_index = start_index + self.num_betas
        betas = current_state[start_index:end_index].view(1,self.num_betas)

        d2v_dt2 = self.vert_acceleration_func(initial_velocity,initial_vertex,current_velocity,current_vertex,primary_shapes_attention,betas,t)
        dv_dt = self.vert_velocity_integrator(current_velocity)

        zeros_vertices = torch.zeros(1,num_vertices*3).float().cuda()
        zeros_vertex_features = torch.zeros(1,num_vertices*32).float().cuda()
        dstate_dt = tuple([torch.cat([self.zeros_num_verts,d2v_dt2.view(1,-1),dv_dt.view(1,-1),zeros_vertices,zeros_vertices,zeros_vertices,self.zeros_betas],1)])
        return dstate_dt
