import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
import numpy as np
np.random.seed(4)
import random
random.seed(4)
from torch.utils.data import Dataset
import os
from os import path as osp
import trimesh
from misc import shape_normalization_transforms
import collections
from MeshProcessor import WaveKernelSignature
from time import time
from omegaconf import OmegaConf
import csv
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import glob
#import cv2
from pytorch3d.transforms import matrix_to_euler_angles,matrix_to_axis_angle

class FilePathDataset(Dataset):
    def __init__(self, root, root_2d):
        self.file_paths = glob.glob(f"{root}/**/keypoints.pt", recursive=True) + glob.glob(f"{root_2d}/**/keypoints.pt", recursive=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return self.file_paths[idx]
    
class Amass(Dataset):
    bm_male = None
    bm_female = None
    bm_neutral = None

    def __init__(self,root,bm_dir,frame_length,max_frames,split=""):
        self.datagroups = list(filter(lambda x:osp.isdir(osp.join(root,x)),os.listdir(root)))
        import random
        self.edges = None
        self.root = root
        self.bm_dir = bm_dir
        self.frame_length = frame_length
        self.max_frames = max_frames
        self.all_motion_dicts = []
        self.all_names = []
        self.bm_fname_male = os.path.join(self.bm_dir, 'smplh/{}/model.npz'.format('male')) 
        self.bm_fname_female = os.path.join(self.bm_dir, 'smplh/{}/model.npz'.format('female')) 
        self.bm_fname_neutral = os.path.join(self.bm_dir, 'smplh/{}/model.npz'.format('neutral')) 
        self.dmpl_fname_male = os.path.join(self.bm_dir, 'dmpls/{}/model.npz'.format('male'))
        self.dmpl_fname_female = os.path.join(self.bm_dir, 'dmpls/{}/model.npz'.format('female'))
        self.dmpl_fname_neutral = os.path.join(self.bm_dir, 'dmpls/{}/model.npz'.format('neutral'))
        comp_device = 'cuda'
        self.bm_male = BodyModel(bm_fname=self.bm_fname_male, num_betas=16, num_dmpls=8, dmpl_fname=self.dmpl_fname_male).to(comp_device)
        self.bm_female = BodyModel(bm_fname=self.bm_fname_female, num_betas=16, num_dmpls=8, dmpl_fname=self.dmpl_fname_female).to(comp_device)
        self.bm_neutral = BodyModel(bm_fname=self.bm_fname_neutral , num_betas=16, num_dmpls=8, dmpl_fname=self.dmpl_fname_neutral).to(comp_device)
        Amass.bm_male = self.bm_male
        Amass.bm_female = self.bm_female
        Amass.bm_neutral = self.bm_neutral

        self.num_betas = 16 ; self.num_dmpls = 8
        self.shapes_per_motion = 1
        self.all_frame_lengths = []
        #self.vid_lengths = []
        #random_betas = np.random.normal(0,0.05,(self.shapes_per_motion,self.num_betas))
        #random_test_betas = 50*random_betas
        self.paths = []
        print("Reading data from")
        # if all_motions:
        npz_files = glob.glob(f"{root}/**/*.npz", recursive=True)
        if split:
            npz_files = filter(lambda x: split in x, npz_files) 
        for motion_f in npz_files:
            print("path: ", motion_f)
            motion_data = dict(np.load(motion_f))
            random_test_betas = np.random.normal(0,3.0,(self.shapes_per_motion,self.num_betas))
            motion_data['betas'] = np.expand_dims(motion_data['betas'],0)  #random_betas
            #motion_data['betas'] = np.array([[-0.1946574, -2.46174753,-1.60554941, 0.7742603,  1.26062725,-0.52041275,-1.50521192, -0.13152285,-1.42397638, 0.00891705,-2.74856981,-1.59050986,0.19627316, 1.04440105,-0.05136906 ,0.20787971]])
            motion_data['test_betas'] = random_test_betas
            self.all_motion_dicts.append(motion_data)
            self.paths.append(motion_f)
            n_frames = motion_data['trans'].shape[0]
            self.all_frame_lengths.append(min(self.max_frames,n_frames))
            frame_rate = motion_data['mocap_framerate']
            num_seconds = n_frames/frame_rate
            name = motion_f.split("/")[-1].split("_poses")[0]
            self.all_names.append(name)
            # for dg in self.datagroups:
            #     motion_dirs = list(filter(lambda x:osp.isdir(osp.join(root,dg,x)),os.listdir(osp.join(root,dg))))
            #     for motion in motion_dirs:
            #         subject_dirs = list(filter(lambda x:osp.isdir(osp.join(root,dg,motion,x)),os.listdir(osp.join(root,dg,motion))))
            #         for subject in subject_dirs:
            #             npz_files = list(filter(lambda x:x.endswith(".npz"),os.listdir(osp.join(root,dg,motion,subject))))
            #             print(npz_files)
            #             for motion_f in npz_files:
            #                 motion_path = osp.join(root,dg,subject,motion_f)
            #                 print("path: ", motion_path)
            #                 motion_data = dict(np.load(motion_path))
            #                 random_test_betas = np.random.normal(0,3.0,(self.shapes_per_motion,self.num_betas))
            #                 motion_data['betas'] = np.expand_dims(motion_data['betas'],0)  #random_betas
            #                 #motion_data['betas'] = np.array([[-0.1946574, -2.46174753,-1.60554941, 0.7742603,  1.26062725,-0.52041275,-1.50521192, -0.13152285,-1.42397638, 0.00891705,-2.74856981,-1.59050986,0.19627316, 1.04440105,-0.05136906 ,0.20787971]])
            #                 motion_data['test_betas'] = random_test_betas

            #                 self.all_motion_dicts.append(motion_data)
            #                 self.paths.append(osp.join(root,dg,subject))
            #                 n_frames = motion_data['trans'].shape[0]
            #                 self.all_frame_lengths.append(min(self.max_frames,n_frames))
            #                 frame_rate = motion_data['mocap_framerate']
            #                 num_seconds = n_frames/frame_rate
            #                 name = motion_f.split("_poses")[0]
            #                 self.all_names.append(name)
        # else:
        #     for dg in self.datagroups:
        #         subject_dirs = list(filter(lambda x:osp.isdir(osp.join(root,dg,x)),os.listdir(osp.join(root,dg))))
        #         for subject in subject_dirs:
        #             # print(os.listdir(osp.join(root,dg,subject)))
        #             npz_files = list(filter(lambda x:x.endswith(".npz"),os.listdir(osp.join(root,dg,subject))))
        #             print(subject)
        #             for motion_f in npz_files:
        #                 motion_path = osp.join(root,dg,subject,motion_f)
        #                 print("path: ", motion_path)
        #                 motion_data = dict(np.load(motion_path))
        #                 random_test_betas = np.random.normal(0,3.0,(self.shapes_per_motion,self.num_betas))
        #                 motion_data['betas'] = np.expand_dims(motion_data['betas'],0)  #random_betas
        #                 #motion_data['betas'] = np.array([[-0.1946574, -2.46174753,-1.60554941, 0.7742603,  1.26062725,-0.52041275,-1.50521192, -0.13152285,-1.42397638, 0.00891705,-2.74856981,-1.59050986,0.19627316, 1.04440105,-0.05136906 ,0.20787971]])
        #                 motion_data['test_betas'] = random_test_betas

        #                 self.all_motion_dicts.append(motion_data)
        #                 self.paths.append(osp.join(root,dg,subject))
        #                 n_frames = motion_data['trans'].shape[0]
        #                 self.all_frame_lengths.append(min(self.max_frames,n_frames))
        #                 frame_rate = motion_data['mocap_framerate']
        #                 num_seconds = n_frames/frame_rate
        #                 name = motion_f.split("_poses")[0]
        #                 self.all_names.append(name)

        total_clips = 0

        for fl in self.all_frame_lengths:
            nc = fl//self.frame_length
            if fl%self.frame_length != 0:
                nc += 1
            total_clips += nc

        self.num_clips = total_clips

    def __len__(self):
        #return len(self.all_endpoints)*self.shapes_per_motion
        return len(self.all_motion_dicts)

    def shuffle(self):
        indices = list(range(len(self.all_motion_dicts)))
        random.shuffle(indices)
        self.all_motion_dicts = [self.all_motion_dicts[i] for i in indices]
        self.all_names = [self.all_names[i] for i in indices]
        self.paths = [self.paths[i] for i in indices]
        #self.vid_lengths = [self.vid_lengths[i] for i in indices]

    def save_max_distance(self):
        np.save(os.path.join(self.root,"maxdistance.npy"),self.max_distance)

    def load_max_distance(self):
        self.max_distance = np.load(os.path.join(self.root,"maxdistance.npy"))

    def __getitem__(self,idx):
        #ep = self.current_all_endpoints[idx]
        #dict_ix = self.current_ep_to_dict_ix[idx]
        #seq_name = self.current_all_names[idx]
        motion_dict = self.all_motion_dicts[idx]
        name = self.all_names[idx]
        #vid_l = self.vid_lengths[idx]
        path = self.paths[idx]
        gender = motion_dict['gender']
        time_length = motion_dict['trans'].shape[0]
        fps = motion_dict['mocap_framerate']

        root_orient = motion_dict['poses'][:, :3]
        pose_body = motion_dict['poses'][:, 3:66]
        pose_hand = motion_dict['poses'][:, 66:]
        trans = motion_dict['trans']

        betas = np.repeat(motion_dict['betas'][:,:self.num_betas][:,np.newaxis], repeats=time_length, axis=1)
        test_betas = np.repeat(motion_dict['test_betas'][:,:self.num_betas][:,np.newaxis], repeats=time_length, axis=1)
        dmpls = motion_dict['dmpls'][:, :self.num_dmpls]

        if gender == 'male':
            g = 0
        elif gender == 'female':
            g = 1
        else:
            g = 2
        #g = random.choice([0,1,2])
        ogs = [x for x in [0,1,2] if x!=g]
        tg = random.choice(ogs)
        return root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,g,tg,name,path #,vid_l

class AmassFeatures(Dataset):
    bm_male = None
    bm_female = None
    bm_neutral = None

    def __init__(self,root,bm_dir,frame_length,max_frames):
        self.datagroups = list(filter(lambda x:osp.isdir(osp.join(root,x)),os.listdir(root)))
        import random
        self.edges = None
        self.root = root
        self.bm_dir = bm_dir
        self.frame_length = frame_length
        self.max_frames = max_frames
        self.all_motion_dicts = []
        self.all_names = []
        self.bm_fname_male = os.path.join(self.bm_dir, 'smplh/{}/model.npz'.format('male')) 
        self.bm_fname_female = os.path.join(self.bm_dir, 'smplh/{}/model.npz'.format('female')) 
        self.bm_fname_neutral = os.path.join(self.bm_dir, 'smplh/{}/model.npz'.format('neutral')) 
        self.dmpl_fname_male = os.path.join(self.bm_dir, 'dmpls/{}/model.npz'.format('male'))
        self.dmpl_fname_female = os.path.join(self.bm_dir, 'dmpls/{}/model.npz'.format('female'))
        self.dmpl_fname_neutral = os.path.join(self.bm_dir, 'dmpls/{}/model.npz'.format('neutral'))
        comp_device = 'cuda'
        self.bm_male = BodyModel(bm_fname=self.bm_fname_male, num_betas=16, num_dmpls=8, dmpl_fname=self.dmpl_fname_male).to(comp_device)
        self.bm_female = BodyModel(bm_fname=self.bm_fname_female, num_betas=16, num_dmpls=8, dmpl_fname=self.dmpl_fname_female).to(comp_device)
        self.bm_neutral = BodyModel(bm_fname=self.bm_fname_neutral , num_betas=16, num_dmpls=8, dmpl_fname=self.dmpl_fname_neutral).to(comp_device)
        Amass.bm_male = self.bm_male
        Amass.bm_female = self.bm_female
        Amass.bm_neutral = self.bm_neutral

        self.num_betas = 16 ; self.num_dmpls = 8
        self.shapes_per_motion = 1
        self.all_frame_lengths = []
        #self.vid_lengths = []
        #random_betas = np.random.normal(0,0.05,(self.shapes_per_motion,self.num_betas))
        #random_test_betas = 50*random_betas
        self.paths = []
        for dg in self.datagroups:
            subject_dirs = list(filter(lambda x:osp.isdir(osp.join(root,dg,x)),os.listdir(osp.join(root,dg))))
            for subject in subject_dirs:
                npz_files = list(filter(lambda x:x.endswith(".npz"),os.listdir(osp.join(root,dg,subject))))

                for motion_f in npz_files:
                    motion_data = dict(np.load(osp.join(root,dg,subject,motion_f)))
                    random_test_betas = np.random.normal(0,3.0,(self.shapes_per_motion,self.num_betas))
                    motion_data['betas'] = np.expand_dims(motion_data['betas'],0)  #random_betas
                    #motion_data['betas'] = np.array([[-0.1946574, -2.46174753,-1.60554941, 0.7742603,  1.26062725,-0.52041275,-1.50521192, -0.13152285,-1.42397638, 0.00891705,-2.74856981,-1.59050986,0.19627316, 1.04440105,-0.05136906 ,0.20787971]])
                    motion_data['test_betas'] = random_test_betas

                    self.all_motion_dicts.append(motion_data)
                    self.paths.append(osp.join(root,dg,subject))

                    n_frames = motion_data['trans'].shape[0]
                    self.all_frame_lengths.append(min(self.max_frames,n_frames))
                    frame_rate = motion_data['mocap_framerate']
                    num_seconds = n_frames/frame_rate
                    name = motion_f.split("_poses")[0]
                    self.all_names.append(name)

        total_clips = 0

        for fl in self.all_frame_lengths:
            nc = fl//self.frame_length
            if fl%self.frame_length != 0:
                nc += 1
            total_clips += nc

        self.num_clips = total_clips

    def __len__(self):
        #return len(self.all_endpoints)*self.shapes_per_motion
        return len(self.all_motion_dicts)

    def shuffle(self):
        indices = list(range(len(self.all_motion_dicts)))
        random.shuffle(indices)
        self.all_motion_dicts = [self.all_motion_dicts[i] for i in indices]
        self.all_names = [self.all_names[i] for i in indices]
        #self.vid_lengths = [self.vid_lengths[i] for i in indices]

    def save_max_distance(self):
        np.save(os.path.join(self.root,"maxdistance.npy"),self.max_distance)

    def load_max_distance(self):
        self.max_distance = np.load(os.path.join(self.root,"maxdistance.npy"))

    def __getitem__(self,idx):
        #ep = self.current_all_endpoints[idx]
        #dict_ix = self.current_ep_to_dict_ix[idx]
        #seq_name = self.current_all_names[idx]
        motion_dict = self.all_motion_dicts[idx]
        name = self.all_names[idx]
        path = self.paths[idx]
        #vid_l = self.vid_lengths[idx]

        gender = motion_dict['gender']
        time_length = motion_dict['trans'].shape[0]
        fps = motion_dict['mocap_framerate']

        root_orient = motion_dict['poses'][:, :3]
        pose_body = motion_dict['poses'][:, 3:66]
        pose_hand = motion_dict['poses'][:, 66:]
        trans = motion_dict['trans']

        betas = np.repeat(motion_dict['betas'][:,:self.num_betas][:,np.newaxis], repeats=time_length, axis=1)
        test_betas = np.repeat(motion_dict['test_betas'][:,:self.num_betas][:,np.newaxis], repeats=time_length, axis=1)
        dmpls = motion_dict['dmpls'][:, :self.num_dmpls]

        if gender == 'male':
            g = 0
        elif gender == 'female':
            g = 1
        else:
            g = 2
        #g = random.choice([0,1,2])
        ogs = [x for x in [0,1,2] if x!=g]
        tg = random.choice(ogs)
        return root_orient,pose_body,pose_hand,trans,betas,test_betas,dmpls,fps,g,tg,name,path #,vid_l
class Animals(Dataset):
    bm_male = None
    bm_female = None
    bm_neutral = None

    def __init__(self,root,bm_dir,frame_length,max_frames):
        self.datagroups = list(filter(lambda x:osp.isdir(osp.join(root,x)),os.listdir(root)))
        import random
        self.edges = None
        self.root = root
        self.frame_length = frame_length
        self.max_frames = max_frames
        self.all_motion_dicts = []
        self.all_names = []

        self.num_betas = 16 ; self.num_dmpls = 8
        self.shapes_per_motion = 1
        self.all_frame_lengths = []
        #self.vid_lengths = []
        #random_betas = np.random.normal(0,0.05,(self.shapes_per_motion,self.num_betas))
        #random_test_betas = 50*random_betas
        for dg in self.datagroups:
            subject_dirs = list(filter(lambda x:osp.isdir(osp.join(root,dg,x)),os.listdir(osp.join(root,dg))))
            for subject in subject_dirs:
                pt_files = list(filter(lambda x:x.endswith(".pt"),os.listdir(osp.join(root,dg,subject))))

                for motion_f in pt_files:
                    motion_data = dict(torch.load(osp.join(root,dg,subject,motion_f)))
                    euler_poses = matrix_to_euler_angles(motion_data['pose'],'XYZ')
                    motion_data['pose'] = euler_poses
                    #motion_data['betas'] = np.expand_dims(motion_data['betas'],0)
                    #motion_data['betas_limbs'] = np.expand_dims(motion_data['betas'],0) 
                    self.all_motion_dicts.append(motion_data)
                    n_frames = motion_data['transl'].shape[0]
                    self.all_frame_lengths.append(min(self.max_frames,n_frames))
                    name = motion_f.split("_motion_poses")[0]
                    self.all_names.append(name)

        total_clips = 0

        for fl in self.all_frame_lengths:
            nc = fl//self.frame_length
            if fl%self.frame_length != 0:
                nc += 1
            total_clips += nc

        self.num_clips = total_clips

    def __len__(self):
        #return len(self.all_endpoints)*self.shapes_per_motion
        return len(self.all_motion_dicts)

    def shuffle(self):
        indices = list(range(len(self.all_motion_dicts)))
        random.shuffle(indices)
        self.all_motion_dicts = [self.all_motion_dicts[i] for i in indices]
        self.all_names = [self.all_names[i] for i in indices]
        #self.vid_lengths = [self.vid_lengths[i] for i in indices]

    def save_max_distance(self):
        np.save(os.path.join(self.root,"maxdistance.npy"),self.max_distance)

    def load_max_distance(self):
        self.max_distance = np.load(os.path.join(self.root,"maxdistance.npy"))

    def __getitem__(self,idx):
        #ep = self.current_all_endpoints[idx]
        #dict_ix = self.current_ep_to_dict_ix[idx]
        #seq_name = self.current_all_names[idx]
        motion_dict = self.all_motion_dicts[idx]
        name = self.all_names[idx]
        #vid_l = self.vid_lengths[idx]

        time_length = motion_dict['transl'].shape[0]

        pose = motion_dict['pose'][:, :]
        trans = motion_dict['transl']

        betas = motion_dict['betas']
        betas_limbs = motion_dict['betas_limbs']
        offset = motion_dict['vertices_offset']

        return pose,trans,betas,betas_limbs,offset,name
    
class D4D(Dataset):
    def __init__(self,root,bm_dir,frame_length,max_frames):
        self.datagroups = list(filter(lambda x:osp.isdir(osp.join(root,x)),os.listdir(root)))
        import random
        self.edges = None
        self.root = root
        self.frame_length = frame_length
        self.max_frames = max_frames
        self.all_motion_dicts = []
        self.all_names = []

        self.num_betas = 16 ; self.num_dmpls = 8
        self.shapes_per_motion = 1
        self.all_verts = []
        self.all_faces = []
        self.all_signatures = []
        self.all_bone_pairs = []
        #self.vid_lengths = []
        #random_betas = np.random.normal(0,0.05,(self.shapes_per_motion,self.num_betas))
        #random_test_betas = 50*random_betas
        self.paths = []
        for dg in self.datagroups:
            subject_dirs = list(filter(lambda x:osp.isdir(osp.join(root,dg,x)),os.listdir(osp.join(root,dg))))
            for subject in subject_dirs:
                obj_files = list(filter(lambda x:x.endswith(".obj"),os.listdir(osp.join(root,dg,subject))))
                obj_files = sorted(obj_files,key=lambda x:int(x.split('.')[0]))
                sub_verts = []
                sub_faces = []
                name = subject 

                self.paths.append(os.path.join(root,dg,subject))
                for motion_f in obj_files:
                    mesh = trimesh.load(os.path.join(root,dg,subject,motion_f),process=False)
                    verts = np.array(mesh.vertices)
                    faces = np.array(mesh.faces)
                    sub_verts.append(np.expand_dims(verts,0))
                    sub_faces.append(np.expand_dims(faces,0))
                self.all_names.append(name)

                if 'bear' in name:
                    sig = np.ones((len(sub_verts),16))*0.25
                elif 'bull' in name:
                    sig = np.ones((len(sub_verts),16))*0.5
                elif 'deer' in name:
                    sig = np.ones((len(sub_verts),16))*0.75
                else:
                    sig = np.ones((len(sub_verts),16))*0.95
                self.all_signatures.append(sig)
                sub_verts = np.concatenate(sub_verts,0)
                sub_faces = np.concatenate(sub_faces,0)

                self.all_verts.append(sub_verts)
                self.all_faces.append(sub_faces)

                bone_face_pairs = []
                kinematic_file = os.path.join(root,dg,subject,subject+".yml")
                kinematics = OmegaConf.load(kinematic_file)
                joints = kinematics["joints"]
                bones = kinematics["bones"]
                kin = kinematics["kinematics"]

                joint_faces = []
                for index,j in joints.items():
                    joint_face = j['face']
                    joint_faces.append(joint_face)

                for index,b in bones.items():
                    joint_0 = b[0] ; joint_1 = b[1]
                    face_0= joints[joint_0]['face']
                    face_1= joints[joint_1]['face']
                    bone_face_pairs.append([face_0,face_1])

                bone_face_pairs = np.array(bone_face_pairs).astype('int')
                bone_pairs = []
                for index,p in kin.items():
                    bone_0 = p[0] ; bone_1 = p[1]
                    b0_face_0 = bone_face_pairs[bone_0][0]
                    b0_face_1 = bone_face_pairs[bone_0][1]
                    b1_face_0 = bone_face_pairs[bone_1][0]
                    b1_face_1 = bone_face_pairs[bone_1][1]
                    bone_pairs.append([b0_face_0,b0_face_1,b1_face_0,b1_face_1])
                bone_pairs = np.array(bone_pairs).astype('int')
                self.all_bone_pairs.append(bone_pairs)

    def __len__(self):
        #return len(self.all_endpoints)*self.shapes_per_motion
        return len(self.all_verts)

    def save_max_distance(self):
        np.save(os.path.join(self.root,"maxdistance.npy"),self.max_distance)

    def load_max_distance(self):
        self.max_distance = np.load(os.path.join(self.root,"maxdistance.npy"))

    def __getitem__(self,idx):
        #ep = self.current_all_endpoints[idx]
        #dict_ix = self.current_ep_to_dict_ix[idx]
        #seq_name = self.current_all_names[idx]
        vertices = self.all_verts[idx]
        faces = self.all_faces[idx]
        name = self.all_names[idx]
        bp = self.all_bone_pairs[idx]
        beta = self.all_signatures[idx]
        path = self.paths[idx]
        return vertices,faces,bp,beta,name,path