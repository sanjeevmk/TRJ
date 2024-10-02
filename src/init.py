import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
from omegaconf import OmegaConf
from collections import namedtuple
from torch.utils.data import DataLoader
from datasets import  Amass,Animals,D4D
import torch.nn as nn
import os
from posing_network import PosingNetwork,OursWKSPosingNetwork,OursWKSPosingNetworkAnimals,NJFWKSPosingNetwork,D4DPosingNetwork
from jacobian_network import JacobianNetwork,JacobianNetworkAnimals,JacobianNetworkD4D
from util_networks import CentroidNormalFC,SeqTransformerJacobians,SeqTransformerTimeJoints,SeqTransformerPoints,VertexFC
from vertex_deformation_network import VertexDeformationNetwork,VertexAccelerationField
from first_order_triangle_ode import AugOde

def initialize_args(config_file):
    config = OmegaConf.load(config_file)

    DataArgs = namedtuple("DataArgs","root bodymodel seq_length max_frames feature_dir")
    TrainingArgs = namedtuple("TrainingArgs","batch epochs lr weight_path preload_path pose_weights feature_weights pose_enc_weights code_size root_zero fixed_time aug_dim")
    AlgorithmArgs = namedtuple("AlgorithmArgs","nsteps stepsize")
    Losses = namedtuple("Losses","mse")
    TestArgs = namedtuple("TestArgs","test testdir src_names tgt_names result_dir method dataset_meshdir motion_name mode nonhuman_mesh nonhuman_features sig")
    LogArgs = namedtuple("LogArgs","dir")

    data_dir = config['datadir']
    bm_dir = config['bodymodel']
    seq_length = config['data']['seq_length']
    max_frames = config['data']['max_frames']
    if "feature_dir" in config['data']:
        feature_dir = config['data']['feature_dir']
    else:
        feature_dir = ""
    data_args = DataArgs(data_dir,bm_dir,seq_length,max_frames,feature_dir)

    batch_size = config['training']['batch']
    epochs = config['training']['epochs']
    code_size = config['training']['code_size']
    lr = config['training']['lr']
    if 'root_zero' in config['training']:
        root_zero = config['training']['root_zero']
    else:
        root_zero = False
    if 'aug_dim' in config['training']:
        aug_dim = config['training']['aug_dim']
    else:
        aug_dim = 0

    if 'fixed_time' in config['training']:
        fixed_time = config['training']['fixed_time']
    else:
        fixed_time = False

    weight_path = config['weight']['path']
    if "preload_path" in config['weight']:
        preload_path = config['weight']['preload_path']
    else:
        preload_path = weight_path
    if 'pose' in config['weight']:
        pose_weights = config['weight']['pose']+"_posing"
        feature_weights = config['weight']['pose']+"_features"
        pose_enc_weights = config['weight']['pose']+"_pose"
    else:
        pose_weights = ""
        feature_weights = ""
        pose_enc_weights = ""
    expt = config['training']['expt']

    if not os.path.exists(config['weight']['directory']):
        os.makedirs(config['weight']['directory'])

    training_args = TrainingArgs(batch_size,epochs,lr,weight_path,preload_path,pose_weights,feature_weights,pose_enc_weights,code_size,root_zero,fixed_time,aug_dim)

    loss_args = Losses(nn.MSELoss())

    test = config['test']['test']
    testdir = config['test']['datadir']
    src_names = config['test']['src_names']
    tgt_names = config['test']['tgt_names']
    result_dir = config['test']['result_dir']
    method = config['test']['method']
    if "nonhuman_mesh" in config['test']:
        nonhuman_mesh = config['test']['nonhuman_mesh']
    else:
        nonhuman_mesh = ""

    if "nonhuman_features" in config['test']:
        nonhuman_features = config['test']['nonhuman_features']
    else:
        nonhuman_features = ""

    if "mode" in config['test']:
        mode = config['test']['mode']
    else:
        mode = "human"

    if "dataset_meshdir" in config['test']:
        dataset_meshdir = config['test']['dataset_meshdir']
    else:
        dataset_meshdir = ""
    if "motion_name" in config['test']:
        motion_name = config['test']['motion_name']
    else:
        motion_name = ""

    if 'sig' in config['test']:
        sig = config['test']['sig']
    else:
        sig = False
    test_args = TestArgs(test,testdir,src_names,tgt_names,result_dir,method,dataset_meshdir,motion_name,mode,nonhuman_mesh,nonhuman_features,sig)

    logdir = config['logging']['directory']
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_args = LogArgs(logdir)
    return data_args,training_args,loss_args,test_args,log_args

def init_amass_sequence_data(root,bm_dir,seq_length=None,max_frames=None):
    dataset = Amass(root,bm_dir,seq_length,max_frames)

    return dataset

def init_animals_sequence_data(root,bm_dir,seq_length=None,max_frames=None):
    dataset = Animals(root,bm_dir,seq_length,max_frames)

    return dataset

def init_d4d_sequence_data(root,bm_dir,seq_length=None,max_frames=None):
    dataset = D4D(root,bm_dir,seq_length,max_frames)

    return dataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.0)
    if classname.find('Conv1D')!=-1:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def init_transformer_network(code_size):
    trans_network = SeqTransformerJacobians(code_size).cuda()
    trans_network.apply(weights_init)
    return trans_network

def init_points_transformer_network(code_size):
    trans_network = SeqTransformerPoints(code_size).cuda()
    trans_network.apply(weights_init)
    return trans_network

def init_joints_encoder_network(code_size):
    joints_encoder = SeqTransformerTimeJoints(64).cuda()
    joints_encoder.apply(weights_init)
    return joints_encoder 

def init_pointnet_network():
    pointnet_network = CentroidNormalFC(6,1).cuda()

    pointnet_network.apply(weights_init)
    return pointnet_network

def init_d3f_pointnet_network():
    #pointnet_network = CentroidNormalFC(2048+6,1).cuda()
    pointnet_network = CentroidNormalFC(6,1).cuda()

    pointnet_network.apply(weights_init)
    return pointnet_network

def init_vertex_pointnet_network():
    vertex_pointnet_network = VertexFC(3,1).cuda()

    vertex_pointnet_network.apply(weights_init)
    return vertex_pointnet_network

def init_posing_network():
    posing_network = PosingNetwork().cuda()
    posing_network.apply(weights_init)
    return posing_network 

def init_d4d_posing_network():
    posing_network = D4DPosingNetwork().cuda()
    posing_network.apply(weights_init)
    return posing_network 

def init_ours_wks_posing_network():
    posing_network = OursWKSPosingNetwork().cuda()
    posing_network.apply(weights_init)
    return posing_network 

def init_njf_wks_posing_network():
    posing_network = NJFWKSPosingNetwork().cuda()
    posing_network.apply(weights_init)
    return posing_network 

def init_ours_wks_posing_network_animals():
    posing_network = OursWKSPosingNetworkAnimals().cuda()
    posing_network.apply(weights_init)
    return posing_network 

def init_jacobian_network(aug_dim=0):
    jacobian_network = JacobianNetwork(aug_dim=aug_dim).cuda()
    jacobian_network.apply(weights_init)
    return jacobian_network 

def init_augode_network(aug_dim):
    augode_network = AugOde(aug_dim).cuda()
    augode_network.apply(weights_init)
    return augode_network 

def init_jacobian_network_d4d():
    jacobian_network = JacobianNetworkD4D().cuda()
    jacobian_network.apply(weights_init)
    return jacobian_network 

def init_jacobian_network_animals():
    jacobian_network = JacobianNetworkAnimals().cuda()
    jacobian_network.apply(weights_init)
    return jacobian_network 

def init_vertex_deformation_network():
    vertex_deformation_network = VertexDeformationNetwork().cuda()
    vertex_deformation_network.apply(weights_init)
    return vertex_deformation_network 

def init_vertex_acceleration_network():
    vertex_acceleration_network = VertexAccelerationField().cuda()
    vertex_acceleration_network.apply(weights_init)
    return vertex_acceleration_network 

def init_optimizers(modules,lr):
    param_list = []
    for m in modules:
        param_list.extend(m.parameters())
    optimizer = torch.optim.Adam(param_list,lr=lr)
    return optimizer


