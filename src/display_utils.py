#from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


custom_colors = ["white","pink","salmon"]
custom_cmap = LinearSegmentedColormap.from_list("mycmap", custom_colors)

cmap = cm.get_cmap('Reds')
gcmap = cm.get_cmap('Greens')
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.2, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
soft_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.00, clip=True)
soft_mapper = cm.ScalarMappable(norm=soft_norm, cmap=gcmap)

def add_loss(writer,name,value,epoch):
    writer.add_scalar(name,value,epoch)
    writer.flush()

def add_mesh(writer,name,vertices,faces,epoch):
    writer.add_mesh(name,vertices,faces=faces,global_step=epoch)

def add_sequence(writer,seq_name,sequence_tensor,seq_faces,epoch):
    dims_present = sequence_tensor.size()[2]
    if dims_present == 2:
        b = sequence_tensor.size()[0]
        n = sequence_tensor.size()[1]
        zeros = torch.zeros(b,n,1)
        sequence_tensor_3d = torch.cat([sequence_tensor,zeros],2)
    else:
        sequence_tensor_3d = sequence_tensor
    faces = seq_faces.repeat(sequence_tensor_3d.size()[0],1,1)
    writer.add_mesh(seq_name,sequence_tensor_3d,faces=faces,global_step=epoch)
    writer.flush()

def write_amass_sequence(root,seq_name,sequence_tensor,seq_faces,start_ix,color,prefix="",v2v_errors=None,v2p_errors=None,soft_errors=None,r90=True,angle=90):
    sequence_tensor_3d = sequence_tensor
    faces = seq_faces.repeat(sequence_tensor_3d.size()[0],1,1)

    sequence_tensor_3d = sequence_tensor_3d.cpu().detach().numpy()
    faces = faces.cpu().detach().numpy()
    import trimesh
    import os

    start = start_ix
    if not os.path.exists(os.path.join(root,seq_name)):
        os.makedirs(os.path.join(root,seq_name))

    for i in range(0,sequence_tensor_3d.shape[0],1):
        verts = sequence_tensor_3d[i,:,:]
        if r90:
            rx = R.from_euler('x',angle,degrees=True).as_matrix()
            verts = verts@rx
        _faces = faces[i,:,:]
        if v2v_errors is None and v2p_errors is None and soft_errors is None:
            vertex_colors = np.zeros(verts.shape)
            vertex_colors[:,:] = color
        elif v2v_errors is not None:
            _v2v_errors = v2v_errors[i]
            v2v_colors = mapper.to_rgba(_v2v_errors.cpu().detach().numpy())
            vertex_colors = v2v_colors[:,:3]
        elif v2p_errors is not None:
            _v2p_errors = v2p_errors[i]
            v2p_colors = mapper.to_rgba(_v2p_errors.cpu().detach().numpy())
            vertex_colors = v2p_colors[:,:3]
        else:
            #print(torch.min(soft_errors),torch.max(soft_errors),torch.mean(soft_errors))
            _soft_errors = soft_errors[i]
            soft_colors = soft_mapper.to_rgba(_soft_errors.cpu().detach().numpy())
            vertex_colors = soft_colors[:,:3]

        _mesh = trimesh.Trimesh(vertices=verts,faces=_faces,vertex_colors=vertex_colors,process=False)

        if prefix:
            _mesh.export(os.path.join(root,seq_name,str(prefix)+"_"+str(start+i)+'.ply'))
        else:
            _mesh.export(os.path.join(root,seq_name,str(start+i)+'.ply'))

def write_sequence(root,seq_name,sequence_tensor,seq_faces,endpoints,color,interp_split):
    dims_present = sequence_tensor.size()[2]
    if dims_present == 2:
        b = sequence_tensor.size()[0]
        n = sequence_tensor.size()[1]
        zeros = torch.zeros(b,n,1)
        sequence_tensor_3d = torch.cat([sequence_tensor,zeros],2)
    else:
        sequence_tensor_3d = sequence_tensor
    faces = seq_faces.repeat(sequence_tensor_3d.size()[0],1,1)

    sequence_tensor_3d = sequence_tensor_3d.cpu().detach().numpy()
    faces = faces.cpu().detach().numpy()
    import trimesh
    import os

    start = endpoints[0]

    #print(start,endpoints[1],sequence_tensor_3d.shape[0],seq_name)
    if not os.path.exists(os.path.join(root,seq_name)):
        os.makedirs(os.path.join(root,seq_name))
    if interp_split > 0:
        with open(os.path.join(root,seq_name,'split.txt'),'w') as f:
            f.write(str(interp_split))

    for i in range(0,sequence_tensor_3d.shape[0],1):
        verts = sequence_tensor_3d[i,:,:]
        #verts /= scale
        #verts -= translation
        bmin = np.min(verts,0)
        bmax = np.max(verts,0)
        diag = np.sqrt(np.sum((bmax-bmin)**2))
        bcenter = np.mean(verts,0)
        #print(diag,bcenter)
        #exit()
        #scale = 1.0/diag
        #translation = -1*bcenter
        _faces = faces[i,:,:]
        if len(color)>0:
            vertex_colors = np.zeros(verts.shape)
            vertex_colors[:,:] = color
        _mesh = trimesh.Trimesh(vertices=verts,faces=_faces,process=False)
        #_mesh = trimesh.Trimesh(vertices=verts,faces=_faces,vertex_colors=[],process=False)

        _mesh.export(os.path.join(root,seq_name,str(start+i)+'.ply'))

def write_multiple_sequences(root,seq_name_list,sequence_tensor_list,seq_faces_list,endpoints_list,color=[],interp_splits=[]):
    for i in range(0,len(sequence_tensor_list)):
        if interp_splits:
            write_sequence(root,seq_name_list[i],sequence_tensor_list[i],seq_faces_list[i],endpoints_list[i],color,interp_split=interp_splits[i])
        else:
            print(seq_name_list[i])
            write_sequence(root,seq_name_list[i],sequence_tensor_list[i],seq_faces_list[i],endpoints_list[i],color,interp_split=0)