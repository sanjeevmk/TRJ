import torch
from torch import nn
import numpy as np
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint

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
        self.zeros_betas = torch.zeros(1,16).float().cuda()
        self.aug_dim = 0 

    def forward(self,t,state):
        current_state = state[0][0]
        start_index = 0
        end_index = start_index+1
        num_faces = current_state[start_index:end_index].int()

        start_index = end_index
        if self.aug_dim == 0:
            end_index = start_index+num_faces*3*3
            transformation_jacobians = current_state[start_index:end_index].view(num_faces,9)
        else:
            end_index = start_index+num_faces*((3*3)+self.aug_dim)
            transformation_jacobians = current_state[start_index:end_index].view(num_faces,9+self.aug_dim)

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
        end_index = start_index + num_faces*32
        cn_feat = current_state[start_index:end_index].view(num_faces,32)

        start_index = end_index
        end_index = start_index + 32
        pose_enc = current_state[start_index:end_index].view(1,32)

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
        djacobians_dt = self.jac_first_order_func(j0,primary_attention,previous_attention,betas,cn_feat,pose_enc,target_j,t)

        zeros_triangle = torch.zeros(1,num_faces*3*3).float().cuda()
        #zeros_cn = torch.zeros(1,num_faces*3*2).float().cuda()
        zeros_cn = torch.zeros(1,num_faces*32).float().cuda()
        zeros_pose = torch.zeros(1,32).float().cuda()
        #zeros_restricted_triangle = torch.zeros(1,num_faces*3*3).float().cuda()
        #zeros_local_feat = torch.zeros(1,num_faces*32).float().cuda()
        dstate_dt = tuple([torch.cat([self.zeros_num_faces,djacobians_dt.view(1,-1),zeros_triangle,zeros_triangle,zeros_triangle,zeros_cn,zeros_pose,zeros_triangle,self.zeros_betas],1)])
        return dstate_dt

class AugOde(nn.Module):
    def __init__(self,aug_dim=0):
        super(AugOde, self).__init__()
        self.fc1 = nn.Linear(9+aug_dim,9) # regular
        #self.fc1 = nn.Linear(9+aug_dim+156,9) # w/ pose
        self.aug_dim = aug_dim

    def forward(self,state,times,num_faces,pose,odeblock):
        #pose_exp = pose.unsqueeze(1).expand(pose.size()[0],num_faces,pose.size()[-1])
        intergrated_solution = odeint(odeblock,state,times,method='euler')[0].squeeze(1)

        start_ix = 1 ; end_ix = start_ix +(num_faces*((3*3)+self.aug_dim))
        jac_first_solution_fetch_indices = torch.from_numpy(np.array(range(start_ix,end_ix))).type(torch.int64).unsqueeze(0).cuda()
        fetch_indices = jac_first_solution_fetch_indices.repeat(len(times),1)
        jac_first_order = torch.gather(intergrated_solution,1,fetch_indices).view(-1,num_faces,(3*3)+self.aug_dim).contiguous()
        #print(jac_first_order)
        #jac_first_order = torch.cat([jac_first_order,pose_exp],-1)
        solution = self.fc1(jac_first_order)
        jac_first_order = solution.view(-1,num_faces,3,3)
        #jac_first_order = jac_first_order[:,:,:9].view(-1,num_faces,3,3)

        return jac_first_order