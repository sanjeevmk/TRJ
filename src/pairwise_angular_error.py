import os
import sys
import numpy as np
import trimesh
from igl import per_face_normals

dire = sys.argv[1]

ours = os.path.join(dire,"eulernjf")
gt = os.path.join(dire,"gt")

fils = os.listdir(ours)
fils = sorted(fils,key=lambda x:int(x.split('.')[0].split('_')[-1]))

for f in fils:
    our_m = trimesh.load(os.path.join(ours,f),process=False)
    gt_m = trimesh.load(os.path.join(gt,f),process=False)
    our_n = per_face_normals(np.array(our_m.vertices),np.array(our_m.faces),np.array([1.0,1.0,1.0]))
    gt_n = per_face_normals(np.array(gt_m.vertices),np.array(gt_m.faces),np.array([1.0,1.0,1.0]))
    a = our_n
    b = gt_n
    inner_product = (a * b).sum(axis=1)
    #print(inner_product.shape)
    cos = inner_product #/ (2 * a_norm * b_norm)
    angle = np.rad2deg(np.arccos(cos))
    print(angle)