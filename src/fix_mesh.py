import pymeshlab
import sys
import numpy as np
import trimesh
import os

def compute_mesh_geo_measures(ms, target_area=np.pi):
    #ms = pymeshlab.MeshSet()
    #mesh = pymeshlab.Mesh(V, F)
    #mesh = pymeshlab.Mesh(V, F)
    #ms.add_mesh(mesh)
    out_dict = ms.get_geometric_measures()
    A = out_dict['surface_area']
    C = np.sqrt( target_area / A )
    out_dict = ms.get_topological_measures()
    return C, out_dict['number_holes']-1

def remove_small_components(v, f,v_idx):
    ## remove small components of the mesh (leave 1 connected component at the end)
    ms   = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(v, f)
    mesh.add_vertex_custom_scalar_attribute(v_idx, 'idx')
    ms.add_mesh(mesh)
    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.compute_selection_by_non_manifold_per_vertex()
    ms.meshing_remove_selected_vertices()
    ms.meshing_re_orient_faces_coherentely()
    # extend selection to non-manifold vertices, this makes easier later stages (parametrization)
    mesh = ms.current_mesh()
    v   = mesh.vertex_matrix()
    f   = mesh.face_matrix()
    idx = mesh.vertex_custom_scalar_attribute_array('idx').astype(np.int64)
    v   = np.array(v.tolist())
    f   = np.array(f.tolist())
    idx = np.array(idx.tolist())
    return v, f, idx

if __name__ == "__main__":
    mesh_dir = sys.argv[1]
    mesh_outdir = sys.argv[2]
    seq_name = os.path.basename(os.path.dirname(mesh_dir))
    if not os.path.exists(mesh_outdir):
        os.makedirs(mesh_outdir)
    obj_files = os.listdir(mesh_dir)
    obj_files = list(filter(lambda x:x.endswith(".obj"),obj_files))
    obj_files = sorted(obj_files,key=lambda x:x.split(".")[0])
    mesh_path = os.path.join(mesh_dir,obj_files[0])
    mesh = trimesh.load(mesh_path,process=False)
    v_idx = np.array(list(range(len(mesh.vertices))))
    v,f,idx = remove_small_components(np.array(mesh.vertices),np.array(mesh.faces),v_idx)
    for objf in obj_files:
        mesh_path = os.path.join(mesh_dir,objf)
        mesh = trimesh.load(mesh_path,process=False)
        new_vertices = np.array(mesh.vertices[idx])
        mesh = trimesh.Trimesh(vertices=new_vertices,faces=f,process=False)
        out_dir = os.path.join(mesh_outdir,seq_name+"/")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        mesh.export(os.path.join(out_dir,objf))