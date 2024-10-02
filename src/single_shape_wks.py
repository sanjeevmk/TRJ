from MeshProcessor import WaveKernelSignature
import sys
import numpy as np
import trimesh
import os

mesh_file = sys.argv[1]
out_file = sys.argv[2]
mesh = trimesh.load(mesh_file,process=False)

verts = np.array(mesh.vertices)
faces = np.array(mesh.faces)

w = WaveKernelSignature(verts, faces, top_k_eig=50)
w.compute()
wk = w.wks
faces_wks = np.zeros((faces.shape[0], wk.shape[1]))
for i in range(3):
    faces_wks += wk[faces[:, i], :]
faces_wks /= 3
np.save(out_file,faces_wks)