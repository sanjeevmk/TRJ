import sys
import os
pair_dir = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]
methods = sys.argv[4:]
methods_str = " ".join(methods)
command = "/home/samk/blender/blender --background --python motion_blend_multiple.py -- " + pair_dir + " " + start + " " + end + " " + methods_str
os.system(command)