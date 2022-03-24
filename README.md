# Synthetic Data

## Running the script

To execute the python script on Blender using a config file

blender -b --python generate.py -- \<filename\>

## Dependencies

This script requires the following nonstandard packages to run
 - bpycv
 - cv2
 - numpy
 - bpy (included with blender)

The script also utilizes the following standard packages
 - abc
 - sys 
 - random
 - os
 - dataclasses
 - typing
 - math
 - time 

## Fender Blenders

If you haven't worked external python scripts and Blender before, there are a few things you need to be sure of:
 - Blender must have access to the python binaries
 - Blender must have access to any python modules imported in the script

Blender will search the PATH environment variable on the machine, and will search its own directory. If all necessary python binaries and modules are accessible via the PATH, then Blender will have no problem executing the script. If you are using a virtual environment that you would like Blender to have access to, replace Blender's python directory wit a directory junction (symbolic link) that redirects to the virtual environment's binaries and modules.
