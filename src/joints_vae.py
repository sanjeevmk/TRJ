import torch.nn as nn
import torch
import numpy as np
import trimesh
import os
import pickle
from torch.autograd.functional import jacobian
import torch.nn.functional as F
from termcolor import colored
import time
import display_utils
import igl
from datetime import datetime,timezone
from PoissonSystem import poisson_system_matrices_from_mesh
from misc import shape_normalization_transforms_pytorch,get_current_time,convert_to_float_tensors,convert_to_float_tensors_from_numpy,convert_to_long_tensors_from_numpy,convert_to_long_tensors
import csv
from torch.autograd import Variable
import random
random.seed(10)
from torch.utils.data import DataLoader
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
from amass_utils import get_sequence,get_clips,get_time_for_clip,get_time_for_jacs,get_atomic_clips,get_pose_of_shape,split_by_indices,get_atomic_clips_by_joints
from timeit import default_timer as timer
import logging
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from math import ceil
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
from misc import compute_errors,align_feet,compute_soft,soft_displacements
from jacobian_network import GetNormals
from torchdiffeq import odeint_adjoint as odeint
from first_order_triangle_ode import TriangleODE,AugOde
from util_networks import CustomPositionalEncoding
from posing_d3f import D3FPosingTrainer as Primary
from MeshProcessor import WaveKernelSignature
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_euler_angles,euler_angles_to_matrix
import math
from skeleton_sequence import sequence_to_skeleton_sequence

class JointsVAE():
    def __init__(self,autoencoder_network):
        self.autoencoder_network = autoencoder_network