import torch
from torch import nn

class TriangleODE(nn.Module):
    def __init__(self,jac_first_order_func):
        super(TriangleODE, self).__init__()
        self.jac_first_order_func = jac_first_order_func 

        #self.prev_t = 0.0
        #self.time_index = 0
        #self.code_size = 16 
        #self.per_time_code_size = 9 
        self.num_betas = 16 
        self.zeros_num_faces = torch.zeros(1,1).float().cuda()
        self.zeros_betas = torch.zeros(1,self.num_betas).float().cuda()

    def forward(self,t,state):
        current_state = state[0][0]
        start_index = 0
        end_index = start_index+1
        num_faces = current_state[start_index:end_index].int()

        start_index = end_index
        end_index = start_index+num_faces*3*3
        transformation_jacobians = current_state[start_index:end_index].view(num_faces,9)

        start_index = end_index
        end_index = start_index + num_faces*3*3
        j0 = current_state[start_index:end_index].view(num_faces,9)

        start_index = end_index
        end_index = start_index + num_faces*3*3
        primary_attention = current_state[start_index:end_index].view(num_faces,9)

        start_index = end_index
        end_index = start_index + num_faces*3*3
        previous_attention = current_state[start_index:end_index].view(num_faces,9)

        start_index = end_index
        end_index = start_index + num_faces*3*3
        target_j = current_state[start_index:end_index].view(num_faces,9)

        #start_index = end_index
        #end_index = start_index + num_faces*32
        #cn = current_state[start_index:end_index].view(num_faces,32)

        start_index = end_index
        end_index = start_index + self.num_betas
        betas = current_state[start_index:end_index].view(1,self.num_betas)

        #start_index = end_index
        #times = current_state[start_index:]
        #xi_jacobians_velocity = xi_jacobians_velocity + (-1*norm_time/spf)*xi_jacobians_2nd + (norm_time/spf)*target_jacobians - norm_time*xi_jacobians_velocity
        djacobians_dt = self.jac_first_order_func(j0,primary_attention,previous_attention,betas,target_j,t)

        zeros_triangle = torch.zeros(1,num_faces*3*3).float().cuda()
        #zeros_cn = torch.zeros(1,num_faces*3*2).float().cuda()
        zeros_cn = torch.zeros(1,num_faces*32).float().cuda()
        zeros_pose = torch.zeros(1,32).float().cuda()
        #zeros_restricted_triangle = torch.zeros(1,num_faces*3*3).float().cuda()
        #zeros_local_feat = torch.zeros(1,num_faces*32).float().cuda()
        dstate_dt = tuple([torch.cat([self.zeros_num_faces,djacobians_dt.view(1,-1),zeros_triangle,zeros_triangle,zeros_triangle,zeros_triangle,self.zeros_betas],1)])
        return dstate_dt
