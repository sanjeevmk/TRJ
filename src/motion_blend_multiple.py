import bpy
import os
import sys
import math
rotation_angle_degrees = 90  # Rotate by 90 degrees about the X-axis
rotation_angle_radians_90 = math.radians(rotation_angle_degrees)
rotation_angle_degrees = 180  # Rotate by 90 degrees about the X-axis
rotation_angle_radians_180 = math.radians(rotation_angle_degrees)


argv = sys.argv
argv = argv[argv.index("--") + 1:] 
# Set the directories where the meshes are stored
pair_dir = argv[0]
start = int(argv[1])
end = int(argv[2])
methods = argv[3:]

primary_method = methods[0]
other_methods = methods[1:]

# Check if there is an object named "Cube" in the scene
if "Cube" in bpy.data.objects:
    # Select the "Cube" object
    bpy.data.objects["Cube"].select_set(True)

    # Delete the selected object
    bpy.ops.object.delete()

for m in other_methods:
    motion_dir = os.path.join(pair_dir,primary_method)

    files = os.listdir(motion_dir)
    files = list(filter(lambda x:x.endswith(".ply"),files))
    files = sorted(files,key=lambda x:int(x.split(".ply")[0].split("_")[1]))[start:end]

    frame_num = 1
    base_mesh = None
    prev_shape_key = None
    for i, file in enumerate(files):
        if file.endswith(".ply") and file.startswith("tgt_"):
            filepath = os.path.join(motion_dir, file)
            bpy.ops.import_mesh.ply(filepath=filepath)
            first_mesh = bpy.context.object
            first_mesh.rotation_euler.x += rotation_angle_radians_90
            selected_meshes = [first_mesh]
            #imported_mesh.select_set(True)

            for om in other_methods:
                other_filepath = os.path.join(pair_dir,om,file)
                bpy.ops.import_mesh.ply(filepath=other_filepath)
                imported_mesh = bpy.context.object
                imported_mesh.select_set(True)
                imported_mesh.rotation_euler.x += rotation_angle_radians_90
                imported_mesh.rotation_euler.z += rotation_angle_radians_180
                translation_distance = -0.75  # Adjust this value as needed
                imported_mesh.location.y += translation_distance
                selected_meshes.append(imported_mesh)
                #empty_object.select_set(True)
                #bpy.context.view_layer.objects.active = empty_object
                #bpy.ops.object.parent_set(type='OBJECT')
            for sm in selected_meshes:
                sm.select_set(True)

            if frame_num==1:
                previous_shape_keys = []
                #base_mesh = empty_object.children[0]
                base_meshes = bpy.context.selected_objects
                #base_mesh = bpy.context.selected_objects[0]
                for bm in base_meshes:
                    #base_mesh.select_set(True)
                    bpy.context.view_layer.objects.active = bm

                    # Add basis shape key
                    shape_key = bm.shape_key_add(name='Basis')
                    # Deselect all vertices of base mesh
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='DESELECT')
                    bpy.ops.object.mode_set(mode='OBJECT')
                    #shape_key.keyframe_insert(data_path='value', frame=frame_num)
                    shape_key.value = 1.0
                    shape_key.keyframe_insert(data_path='value', frame=frame_num)
                    prev_shape_key = shape_key
                    previous_shape_keys.append(prev_shape_key)
                    bm.select_set(False)
            else:
                #deformation_mesh = empty_object.children[0]
                # Set deformation mesh vertices as shape key
                deformation_meshes = bpy.context.selected_objects
                #deformation_mesh = bpy.context.selected_objects[0]
                # Update base mesh transform
                for m_ix,deformation_mesh in enumerate(deformation_meshes):
                    def_vertices = []
                    for ix in range(len(deformation_mesh.data.vertices)):
                        def_vertices.append(deformation_mesh.data.vertices[ix].co)

                    base_meshes[m_ix].select_set(True)
                    bpy.context.view_layer.objects.active = base_meshes[m_ix]
                    # Add new shape key
                    shape_key = base_meshes[m_ix].shape_key_add(name=str(frame_num))
                    # Update base mesh vertices
                    for ix in range(len(base_meshes[m_ix].data.vertices)):
                        shape_key.data[ix].co = def_vertices[ix]

                    shape_key.value = 1.0
                    shape_key.keyframe_insert(data_path='value', frame=frame_num)

                    print(len(previous_shape_keys),m_ix)
                    # Set previous shape key value to 0.0 and keyframe it
                    previous_shape_keys[m_ix].value = 0.0
                    previous_shape_keys[m_ix].keyframe_insert(data_path='value', frame=frame_num)

                    #prev_shape_key = shape_key
                    # Update shape key vertices
                    #shape_key.data.foreach_set("co", [c for v in base_mesh.data.vertices for c in v.co])

                    bpy.data.objects.remove(deformation_mesh)

            frame_num +=1

# Set interpolation mode for all shape keys
for bm in base_meshes:
    for shape_key in bm.data.shape_keys.key_blocks:
        shape_key.interpolation = 'KEY_BSPLINE'
        #shape_key.handle_left_type = 'VECTOR'
        #shape_key.handle_right_type = 'VECTOR'

for bm in base_meshes:
    # Keyframe the shape keys to animate them
    for i, shape_key in enumerate(bm.data.shape_keys.key_blocks):
        # Set the value of the shape key to 1.0 and keyframe it
        shape_key.value = 1.0
        shape_key.keyframe_insert(data_path='value', frame=i+1)

        # Set the value of all other shape keys to 0.0 and keyframe them
        for j, other_key in enumerate(bm.data.shape_keys.key_blocks):
            if i != j:
                other_key.value = 0.0
                other_key.keyframe_insert(data_path='value', frame=i+1)
'''
# Keyframe the shape key value at the current frame
for fix in range(frame_num):
    for kix,shape_key in enumerate(base_mesh.data.shape_keys.key_blocks):
        if fix!=kix:
            shape_key.value = 0
        else:
            shape_key.value = 1
        shape_key.keyframe_insert(data_path='value', frame=frame_num)
'''

# Get the second-to-last directory name
second_last_dir = os.path.basename(os.path.dirname(motion_dir.strip("/")))

# Get the last directory name
last_dir = os.path.basename(motion_dir.strip("/"))

output_path = os.path.abspath(os.path.join(motion_dir,second_last_dir+"_"+last_dir+".blend"))
print(output_path)
#print(second_last_dir,last_dir)
# Set the output file path for the .blend file
#output_path = "/path/to/output.blend"
# Save the .blend file
bpy.ops.wm.save_as_mainfile(filepath=output_path)
