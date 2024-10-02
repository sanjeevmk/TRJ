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
np.save(out_file,wk)