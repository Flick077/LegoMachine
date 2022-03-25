# Synthetic Data

## To the soon to be frustrated reader

I apologize for the poor quality of interfacing with this script... that's all I got. Good luck!

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

## LDView and LDraw

If you have LDView and LDraw on your machine and would like to automatically generate stl files for parts that are requested, then you will need to create a file titled gen.inf in the same directory as the generator script, and add the following lines of text.

    ldraw=/path/to/ldraw/parts/
    ldview=/path/to/ldview/excutables/
