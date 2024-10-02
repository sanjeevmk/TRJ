import trimesh
import yaml
import numpy as np
import sys
import torch
from collections import OrderedDict

kinematic_path = "../data/kinematics.yml"
with open(kinematic_path, 'r') as file:
    data = yaml.safe_load(file)

joints = data['joints']
bones = data['bones']
kinematics = data['kinematics']

def create_dual_tetrahedron_bone(vertex1, vertex5, direction,random_direction, radius,end_radius=0.02):
    bone_length = np.linalg.norm(direction)
    triangle_size = calculate_triangle_size(bone_length)
    forward_direction = direction / bone_length
    start = vertex1 + radius * forward_direction
    end = vertex5 - end_radius * forward_direction
    direction = end - start
    bone_length = np.linalg.norm(direction)
    forward_direction = direction / bone_length

    # Calculate the centroid of the triangle
    centroid = start + 0.1 * direction

    # Generate a random vector perpendicular to the forward direction
    perpendicular_vector = np.cross(forward_direction, random_direction)
    perpendicular_vector /= np.linalg.norm(perpendicular_vector)
    perpendicular_vector *= triangle_size

    # Rotate the perpendicular vector by 120 degrees and 240 degrees
    rotation_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(120), forward_direction)
    perpendicular_vector120 = np.dot(rotation_matrix[:3, :3], perpendicular_vector)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(240), forward_direction)
    perpendicular_vector240 = np.dot(rotation_matrix[:3, :3], perpendicular_vector)

    # Calculate the vertices of the triangle
    vertex2 = centroid + perpendicular_vector
    vertex3 = centroid + perpendicular_vector120
    vertex4 = centroid + perpendicular_vector240

    tetrahedron_vertices = [
        start,
        vertex2,
        vertex3,
        vertex4,
        end
    ]

    tetrahedron_faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
        [4, 1, 2],
        [4, 2, 3],
        [4, 3, 1]
    ]

    mesh = trimesh.Trimesh(vertices=tetrahedron_vertices, faces=tetrahedron_faces, process=False)

    return mesh

def calculate_triangle_size(bone_length):
    # Adjust the scaling factor as desired
    scaling_factor = 0.1
    return scaling_factor * bone_length

def mesh_to_centroids(vertices,faces):
    centroids = vertices[faces].mean(axis=1)

    excluded_joints = ['head','l_hand','r_hand','l_foot','r_foot']
    list_of_tuples = [(key, joints[key]) for key in joints] # if key not in excluded_joints]
    ord_joints = OrderedDict(list_of_tuples)
    # Step 5: Create the skeleton mesh
    # Create joint spheres
    pose_centroids = []
    for joint_id, joint_info in ord_joints.items():
        face_index = joint_info['face']
        sphere_position = centroids[face_index]
        pose_centroids.append(sphere_position)

    pose_centroids = torch.cat(pose_centroids,0)
    return pose_centroids



def mesh_to_skeleton_mesh(vertices,faces,random_direction):
    centroids = vertices[faces].mean(axis=1)
    sphere_radius = 0.02  # Adjust the radius as desired
    head_radius = 0.05  # Adjust the radius as desired

    # Step 4: Compute the bone directions based on kinematics
    bone_directions = []
    for bone_id, bone_info in bones.items():
        joint_indices = bone_info
        start_joint = joints[joint_indices[0]]
        end_joint = joints[joint_indices[1]]
        start_centroid = centroids[start_joint['face']]
        end_centroid = centroids[end_joint['face']]
        direction = end_centroid - start_centroid
        bone_directions.append(direction)

    excluded_joints = ['head','l_hand','r_hand','l_foot','r_foot']
    # Step 5: Create the skeleton mesh
    skeleton_mesh = trimesh.Trimesh()
    # Create joint spheres
    for joint_id, joint_info in joints.items():
        face_index = joint_info['face']
        sphere_position = centroids[face_index]
        if joint_info['name'] not in excluded_joints:
            sphere = trimesh.creation.uv_sphere(radius=sphere_radius)
            sphere.apply_translation(sphere_position)
            skeleton_mesh = trimesh.util.concatenate([skeleton_mesh, sphere])

    # Create joint spheres and bone tetrahedrons
    for bone_id, bone_info in bones.items():
        joint_indices = bone_info
        start_joint = joints[joint_indices[0]]
        end_joint = joints[joint_indices[1]]
        start_position = centroids[start_joint['face']]
        end_position = centroids[end_joint['face']]
        direction = bone_directions[bone_id]
        if joint_indices[1]==12:
            bone = create_dual_tetrahedron_bone(start_position, end_position, direction,random_direction,sphere_radius,end_radius=head_radius)
        else:
            bone = create_dual_tetrahedron_bone(start_position, end_position, direction,random_direction,sphere_radius)
        skeleton_mesh = trimesh.util.concatenate([skeleton_mesh, bone])

    # Create a larger sphere for the head
    head_joint = joints[12]  # Assuming the head joint is at index 0
    head_position = centroids[head_joint['face']]
    head_sphere = trimesh.creation.uv_sphere(radius=head_radius)
    head_sphere.apply_translation(head_position)
    skeleton_mesh = trimesh.util.concatenate([skeleton_mesh, head_sphere])

    return np.array(skeleton_mesh.vertices),np.array(skeleton_mesh.faces)

def sequence_to_skeleton_sequence(sequence,faces):
    skel_f = None
    skel_vertices = []
    random_direction = np.random.randn(3)
    for i in range(sequence.shape[0]):
        sk_v,sk_f = mesh_to_skeleton_mesh(sequence[i],faces,random_direction)
        skel_f = sk_f
        skel_vertices.append(torch.from_numpy(sk_v).unsqueeze(0).float().cuda())

    skel_vertices = torch.cat(skel_vertices,0)
    skel_f = torch.from_numpy(skel_f).long().cuda()

    return skel_vertices,skel_f

def sequence_to_centroids(sequence,faces):
    skel_f = None
    skel_vertices = []
    random_direction = np.random.randn(3)
    for i in range(sequence.shape[0]):
        sk_v = mesh_to_centroids(sequence[i],faces)
        skel_vertices.append(sk_v.unsqueeze(0).float().cuda())

    skel_vertices = torch.cat(skel_vertices,0)

    return skel_vertices
