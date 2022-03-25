# Synthetic Data

## To the soon to be frustrated reader

I apologize for the poor quality of interfacing with this script... that's all I got. Good luck!

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

## Running the script

To execute the python script on Blender using a config file:

    blender -b --python generate.py -- example.cfg
    
### Config Files

The config files for this script are intended to make manipulation of the dataset parameters more flexible, but the parsing algorithm is not very robust (it doesn't even check the file extension at the moment XD). Config files are text files in which each line is a key-value pair separated by an "=". See below for a list of keys currently recommended for use:
 - **dataset**: the name of the dataset. This can be a new dataset or a previously existing one, the script is capable of appending classes to the yaml files when necessary. Defaults to "data".
 - **parts**: a comma-separated list of part numbers for the lego models to be included in the dataset. 
 - **engine**: the engine blender should use to render images. Optional values for this key are: *BLENDER_EEVEE*, or _CYCLES_
 - **size**: the number of images to generate. Defaults to 1000
 - **capacity**: the maximum number of legos to include in a single image.
 - **gravity**: boolean variable with values _on_ or _off_. Enables gravity simulation prior to rendering the image in order to add realism. Defaults to _on_.
 - **split**: controls the split between training, validation, and test data. Should be a comma-separated list of three numeric values in the order _train_, _val_, _test_. The values will automatically be normalized, but will default to 70% train, 20% validate, and 10% test (i.e., "split=7, 2, 1").

An [example config file](SyntheticData/example.cfg) is available for viewing in the SyntheticData folder. 

## Fender Blenders

If you haven't worked external python scripts and Blender before, there are a few things you need to be sure of:
 - Blender must have access to the python binaries
 - Blender must have access to any python modules imported in the script

Blender will search the PATH environment variable on the machine, and will search its own directory. If all necessary python binaries and modules are accessible via the PATH, then Blender will have no problem executing the script. If you are using a virtual environment that you would like Blender to have access to, replace Blender's python directory wit a directory junction (symbolic link) that redirects to the virtual environment's binaries and modules.

## LDView and LDraw

If you have LDView and LDraw on your machine and would like to automatically generate stl files for parts that are requested, then you will need to create a file titled gen.inf in the same directory as the generator script, and add the following lines of text.

    ldraw=/path/to/ldraw/parts/
    ldview=/path/to/ldview/excutables/
