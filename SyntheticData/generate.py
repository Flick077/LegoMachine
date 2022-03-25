import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

import bpy
import bpycv
import os
import math
import time

import cv2
import numpy as np

# TODO
#   - add stl conveyor background
#       > -19m on x-axis for center
#       > 13m displacement on y-axis to join the pieces
#   - add dirty texture to legos

# a list of the most common lego colors from the rebrickable site
lego_colors = {
    'black': '0x05131d',
    'white': '0xffffff',
    'medium stone gray': '0xa0a5a9',
    'dark stone gray': '6c6e68',
    'red': 'c91a09',
    'yellow': '0xf2cd37',
    'blue': '0x0055bf'
}


@dataclass
class Argument(ABC):
    name: str
    flag: str
    shortcut: str = None
    default: Union[str, list] = None
    values: List[str] = None
    help_msg: str = None
    is_required: bool = False

    def parse(self, args: List[str], kwargs: dict):
        kwargs[self.name] = self.default if self.name not in kwargs else kwargs[self.name]
        flag = None
        if self.flag in args:
            flag = self.flag
        elif self.shortcut and self.shortcut in args:
            flag = self.shortcut
        if flag is not None:
            self._parse(flag, args, kwargs)
        else:
            self._no_parse(kwargs)

    @abstractmethod
    def _parse(self, flag: str, args: List[str], kwargs: dict):
        pass

    def _no_parse(self, kwargs: dict):
        kwargs[self.name] = self.default if self.name not in kwargs else kwargs[self.name]

    def validate(self, kwargs: dict):
        if self.name not in kwargs:
            kwargs[self.name] = self.default
        else:
            self._validate(kwargs)

    @abstractmethod
    def _validate(self, kwargs: dict):
        pass

    def value_error(self, msg: str = None, value: str = None):
        if msg is None:
            msg = ' '.join(filter(
                lambda m: len(m) > 0,
                [f'Argument {self.name} ({self.flags()}) received an invalid value', value if value else '']))
        msg += f'\nUse --help or -h followed by {self.flags()}, or {self.name} for more information'
        raise ArgumentError(msg)

    def flags(self) -> str:
        return ', '.join(filter(lambda flag: len(flag) > 0, [self.flag, self.shortcut if self.shortcut else '']))

    def help(self):
        if self.help_msg is None:
            self.help_msg = f'Argument: {self.name}\n'
            self.help_msg += f'\tFlags: {self.flags()}'

        print(self.help_msg)


class BooleanArgument(Argument):
    def __init__(self, name: str, flag: str, shortcut: str):
        super().__init__(name, flag, shortcut)

    def _parse(self, flag: str, args: List[str], kwargs: dict):
        kwargs[self.name] = True

    def _no_parse(self, kwargs: dict):
        kwargs[self.name] = False

    def _validate(self, kwargs: dict):
        if kwargs[self.name] in ['True', 'False']:
            kwargs[self.name] = bool(kwargs[self.name])
        if kwargs[self.name] not in [True, False, 'True', 'False']:
            self.value_error(value=kwargs[self.name])

    def help(self):
        if self.help_msg is None:
            self.help_msg = f'Argument: {self.name}\n'
            self.help_msg += f'\tFlags: {self.flags()}\n\n'
            self.help_msg += f'Include this flag to enable or disable its features during execution'
        super().help()


class SingleArgument(Argument):
    def _parse(self, flag: str, args: List[str], kwargs: dict):
        idx = args.index(flag) + 1
        if idx >= len(args) or args[idx].startswith('-'):
            flags = [self.flag, self.shortcut if self.shortcut else '']
            self.value_error(
                f'Argument {self.name} ({self.flags()}) expected a following token, but did not receive one')
        kwargs[self.name] = args[idx]

    def _validate(self, kwargs: dict):
        if self.values is not None and kwargs[self.name] not in self.values:
            self.value_error(value=kwargs[self.name])

    def help(self):
        if self.help_msg is None:
            self.help_msg = f'Argument: {self.name}\n'
            self.help_msg += f'\tFlags: {self.flags()}\n'
            if self.values:
                self.help_msg += f'\tValues: {",".join(self.values)}\n\n'
            self.help_msg += f'Include this argument flag followed by a value to set the parameter for execution'
        super().help()


class ConfigArgument(SingleArgument):
    def __init__(self):
        super().__init__('config', '--config', '-c')

    def _parse(self, flag: str, args: List[str], kwargs: dict):
        idx = args.index(flag) + 1
        if idx >= len(args) or args[idx].startswith('-'):
            self.value_error()
        _parse_config(args[idx], kwargs)


class ResolutionArgument(SingleArgument):
    def __init__(self):
        super().__init__('resolution', '--resolution', '-res', default='640x640')

    def _validate(self, kwargs: dict):
        try:
            res = kwargs[self.name].split('x')
            if len(res) != 2 or not float(res[0]) >= 0 or not float(res[1]) >= 0:
                self.value_error()
        except Exception as e:
            self.value_error()


class MultiArgument(Argument):
    def _parse(self, flag: str, args: List[str], kwargs: dict):
        idx = args.index(flag) + 1
        if idx >= len(args) or args[idx].startswith('-'):
            flags = [self.flag, self.shortcut if self.shortcut else '']
            self.value_error(
                f'Argument {self.name} ({self.flags()}) expected a following token, but did not receive one')
        kwargs[self.name] = []
        for arg in args[idx:]:
            if arg.startswith('-'):
                break
            kwargs[self.name].append(arg)

    def _validate(self, kwargs: dict):
        if self.values is not None and kwargs[self.name] not in self.values:
            self.value_error(value=kwargs[self.name])


class SetSplitArgument(MultiArgument):
    def __init__(self):
        super().__init__(name='split', flag='--split', default=['0.7', '0.2', '0.1'])

    def _validate(self, kwargs: dict):
        # store default if not yet set
        if self.name not in kwargs:
            kwargs[self.name] = list(map(lambda s: float(s), self.default))
            return
        # check type
        val = kwargs[self.name]
        if type(val) is not list or len(val) != 3:
            self.value_error(value=val)
        # normalize split
        val = np.array(list(map(lambda s: float(s), val)))
        kwargs[self.name] = list(val / val.sum())


def parse_args(arguments: List[Argument], argv: List[str]) -> dict:
    argv = _remove_blender_args(argv)
    kwargs = dict()
    exit_signal = {'exit': True}
    h_flag = None
    if '--help' in argv:
        h_flag = '--help'
    elif '-h' in argv:
        h_flag = '-h'
    if h_flag is not None:
        idx = argv.index(h_flag) + 1
        if idx >= len(argv):
            print(f'Use the {h_flag} flag followed by an argument flag to see more information about that argument')
            return exit_signal
        arg_flag = argv[idx]
        for argument in arguments:
            if arg_flag in argument.flags().split(','):
                argument.help()
                return exit_signal
    if not argv[0].startswith('-'):
        # config file
        _parse_config(argv[0], kwargs)
        arguments = list(filter(lambda a: a.name != 'config', arguments))
    for argument in arguments:
        argument.parse(argv, kwargs)
    if '--config-strict' not in argv:
        _ensure_args(arguments, kwargs)
    _parse_inf(kwargs)
    return kwargs


def _ensure_args(arguments: List[Argument], kwargs: dict, error_on_all=False):
    """
    Perform data validation on the keyword arguments, and remove any invalid entries
    :param kwargs: keyword arguments for validation
    """
    for argument in arguments:
        try:
            argument.validate(kwargs)
        except ArgumentError as e:
            if error_on_all or argument.is_required:
                # propagate error when appropriate
                raise e
            else:
                # point invalid attribute to default (None if not specified)
                print(f'Argument {argument.name} ({argument.flags()}) could not handle given value '
                      f'{kwargs[argument.name]}. Setting to default value of {argument.default}')
                kwargs[argument.name] = argument.default


def _parse_config(config_file: str, kwargs: dict):
    """
    Set script parameters based on given config file.
    :param config_file: file path of the config file
    :param kwargs: dictionary to add script parameters to
    :return: None
    """
    with open(config_file, 'r') as cfg:
        line_num = 0
        line = 'empty'
        try:
            while len(line) > 0:
                line_num = 1
                line = cfg.readline()
                # check for end of file
                if not line:
                    break
                # check for keyword declaration
                if '=' in line:
                    key, line = line.strip().split('=')
                    if key == 'resolution':
                        kwargs['res_x'], kwargs['res_y'] = line.split('x')
                    else:
                        kwargs[key] = line.split(',') if ',' in line else line
                elif ',' in line:
                    kwargs[key].extend(line.split(','))
        except Exception as e:
            # propagate exception with additional information
            raise ConfigFileError(config_file, line_num, line, e)


def _parse_inf(kwargs: dict):
    """
    Reads the gen.inf file to find supporting executables and data stores.
    """
    if not os.path.exists('gen.inf'):
        raise Exception('Cannot execute without gen.inf file')
    with open('gen.inf', 'r') as inf:
        for line in inf.readlines():
            if '=' in line:
                key, val = line.split('=', 1)
                kwargs[key] = val


def _remove_blender_args(args: List[str]) -> List[str]:
    """
    Separates arguments for script from arguments for blender.

    :param args: The arguments passed through the terminal (sys.argv)
    :return: Only the arguments intended for the called python script
    """
    return args[args.index('--') + 1:] if '--' in args else args


def remove_objects(types: List[str] = None, keep: List[bpy.types.Object] = None):
    """
    Removes all objects of a given type from the active blender instance. Will remove all objects if not specified.

    :param types: the type of object to remove (e.g. 'MESH', 'LIGHT', etc.)
    :param keep: any blender objects that should not be removed
    """
    if keep is None:
        keep = []
    for obj in bpy.data.objects:
        if (obj.type in types or types is None) and obj not in keep:
            bpy.data.objects.remove(obj)


def add_background(file: str, kwargs: dict) -> bpy.types.Object:
    """
    Imports an image into the file as a plane to serve as the background and the resting surface of the legos.

    :param file: the file path of the background image
    :param kwargs: the keyword arguments parsed from config and terminal
    """
    bpy.ops.import_image.to_plane(files=[{"name": file}],
                                  directory="scenes\\",
                                  relative=True)
    if kwargs['gravity'] == 'on':
        bpy.ops.rigidbody.objects_add(type='PASSIVE')
    bg = bpy.context.active_object
    bg.scale = (100,) * 3
    bg.rotation_euler = (0, 0, 0)
    bg.location = (0, 7.5, 0)
    return bg


def add_lego(part: str, kwargs: dict) -> bpy.types.Object:
    """
    Adds a lego object to the blender scene.

    :param part: the model number of the lego part to add
    :param kwargs: the keyword arguments parsed from config file and terminal
    :return: The reference to the lego object.
    """
    # initialize instance id if necessary
    if 'inst_id' not in kwargs:
        kwargs['inst_id'] = 1
    # add mesh to scene
    bpy.ops.import_mesh.stl(filepath=os.path.join(kwargs['stl_dir'], f'{part}.stl'))
    lego = bpy.context.active_object
    # add object's instance id for bpycv
    lego["inst_id"] = kwargs['inst_id']
    kwargs['inst_id'] += 1
    # add randomized material to the lego
    lego.active_material = _randomized_material()
    # randomize location and orientation
    lego.location = (random.random() * 40 - 20, random.random() * 40 - 20, 5)
    lego.rotation_euler = tuple(math.radians(random.randrange(360)) for _ in range(3))
    # enable necessary constraints for gravity simulation
    if kwargs["gravity"] == 'on':
        bpy.ops.rigidbody.objects_add(type='ACTIVE')
        if 'lego_area' not in kwargs:
            # use cam data to calculate viewable area
            cam = bpy.context.scene.camera
            ymin = cam.location[1] + math.tan(cam.rotation_euler[0] - cam.data.angle / 2) * cam.location[2]
            ymax = cam.location[1] + math.tan(cam.rotation_euler[0] + cam.data.angle / 2) * cam.location[2]
            xdis = cam.location[2] * math.tan(cam.data.angle / 2)
            # cam.location[2] * math.tan(cam.data.angle / 2) / math.cos(cam.rotation_euler[0] - cam.data.angle / 2)
            xmax = cam.location[0] + xdis
            xmin = cam.location[0] - xdis
            # TODO: better fit of view area
            # wmin = 2 * cam.location[2] * math.tan(cam.data.angle / 2) / math.cos(
            #     cam.rotation_euler[0] - cam.data.angle / 2)
            # wmax = 2 * cam.location[2] * math.tan(cam.data.angle / 2) / math.cos(
            #     cam.rotation_euler[0] + cam.data.angle / 2)
            kwargs['lego_area'] = (xmin, ymin, xmax, ymax)
    return lego


def _randomized_material(**kwargs) -> bpy.types.Material:
    """
    Generate a domain-randomized material for a lego blender object
    """
    # TODO: add optional fixed keyword arguments
    # TODO: add data set config file
    # TODO: texture randomization
    color_name, color = _random_color()
    mat = bpy.data.materials.new(color_name)
    mat.use_nodes = True
    principled = mat.node_tree.nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['Roughness'].default_value = 0.0
    principled.inputs['Specular'].default_value = 1.0
    return mat


def _random_color() -> (str, Tuple):
    """
    Selects a random lego color
    :return: The color name and the rgba tuple value for the color
    """
    key = random.choice(list(lego_colors.keys()))
    return key, _hex_to_rgba(lego_colors[key])


def _hex_to_rgba(hex_str: str) -> Tuple:
    """
    Converts a hex color string to an rgba tuple
    :param hex_str: A string of form 'rrggbb', 'rrggbbaa', '0xrrggbb', or '0xrrggbbaa'
    :return: the equivalent tuple
    """
    if hex_str[:2] == '0x':
        hex_str = hex_str[2:]
    if len(hex_str) == 6:
        return (*[int(hex_str[s:s + 2], 16) / 255.0 for s in range(0, 6, 2)], 1.0)
    if len(hex_str) == 8:
        return tuple(int(hex_str[s:s + 2], 16) / 255.0 for s in range(0, 8, 2))
    raise ValueError("hex_str should be of form 'rrggbb', 'rrggbbaa', '0xrrggbb', or '0xrrggbbaa'")


def ensure_dataset(kwargs: dict):
    """
    Ensures the directory structure for the dataset is prepared for data generation

    :param kwargs: the keyword arguments parsed from the config file and terminal
    :return:
    """
    # ensure directory structure
    dset = kwargs['dataset']
    _ensure_dir('datasets')
    _ensure_dir(os.path.join('datasets', dset))
    for dir in ['images', 'labels']:
        _ensure_dir(os.path.join('datasets', dset, dir))
        for fdr in ['train', 'test', 'val']:
            _ensure_dir(os.path.join('datasets', dset, dir, fdr))

    # ensure YAML file
    yaml = os.path.join('datasets', dset, f'{dset}.yaml')
    cnums = []
    if not os.path.exists(yaml):
        cnums = kwargs['parts']
    else:
        from pandas import unique
        with open(yaml, 'r') as file:
            s = ''
            line = file.readline()
            while line:
                if 'names' in line:
                    s = line[line.find('['):].strip()
                elif len(s) > 0:
                    s += line.strip()
                if ']' in s:
                    break
                line = file.readline()
            cnums = list(map(lambda y: y.strip(" \'"), s.strip('[]').split(',')))
            cnums.extend(kwargs['parts'])
            cnums = list(unique(cnums))
    with open(yaml, '+w') as file:
        # point to directories
        cwd = os.getcwd()
        for fdr in ['train', 'test', 'val']:
            file.write(f'{fdr}: {os.path.join(cwd, "datasets", dset, "images", fdr)}\n')
        file.write('\n')

        # class information
        file.write(f'nc: {len(cnums)}\n')
        file.write(f'names: {cnums}')
    kwargs['cnums'] = cnums


def _ensure_dir(path: str):
    """
    Checks for the existence of a path name, and creates a directory if there is none

    :param path: the directory to ensure existence of
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)


def save_to_set(cnums: list, fdr: str, kwargs: dict):
    """
    Saves the current blender file to the dataset directory in compliance with YOLO's specifications

    :param cnums: the class numbers of the added parts
    :param fdr: the subfolder of the dataset directory to place the images and labels in (i.e. 'train', 'val', 'test')
    :param kwargs: the keyword arguments parsed from commandline and config
    """
    # generate unique file name
    fname = str(time.time()).replace('.', '')

    # find bounding boxes
    # WARN: may need to set frame to 100 if gravity is enabled
    bpy.context.scene.frame_set(100)
    img = bpycv.render_data()
    unique, _idxs = np.unique(img["inst"], return_inverse=True)
    idxs = _idxs.reshape(img["inst"].shape)
    boxes = np.zeros((len(cnums), 4))
    render = bpy.context.scene.render
    popped = 0
    for c in range(len(cnums)):
        # find all pixels of the instance id
        hits = np.argwhere(idxs == c + 1)
        if len(hits) == 0:
            # object off screen during render
            popped += 1
            cnums.pop(c)
            continue
        # find x and y bounds
        mins = hits.min(axis=0)
        maxs = hits.max(axis=0)
        # locate center
        i = c-popped
        boxes[i][0] = (mins[1] + maxs[1]) / 2
        boxes[i][1] = (mins[0] + maxs[0]) / 2
        # width and height
        boxes[i][2] = (maxs[1] - mins[1])
        boxes[i][3] = (maxs[0] - mins[0])
        boxes[i] /= np.array([render.resolution_x, render.resolution_y, ] * 2)

    # write txt file for bounding box
    with open(os.path.join('datasets', kwargs['dataset'], 'labels', fdr, f'{fname}.txt'), '+w') as yaml:
        for item in range(len(cnums)):
            yaml.write(' '.join([str(cnums[item]), *(map(lambda f: str(f), boxes[item]))]) + '\n')

    # save image
    cv2.imwrite(os.path.join('datasets', kwargs['dataset'], 'images', fdr, f'{fname}.png'), img["image"][..., ::-1])


class ConfigFileError(Exception):
    def __init__(self, filename: str, line_num: int, line: str, e: Exception):
        msg = f'Invalid config file {filename} generated an exception at line {line_num}. Line contents:\n'
        msg += f'\t{line}\n'
        msg += f'Original exception message: {str(e)}'
        super().__init__(msg)


class ConfigValueError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class ArgumentError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


def main(kwargs: dict):
    # ensure dataset directory is prepared for data
    ensure_dataset(kwargs)

    # temporarily add LDView to environment path
    if 'ldview' in kwargs:
        os.environ["PATH"] += os.pathsep + kwargs['ldview']

    # ensure all stl files are available
    dir = kwargs['stl_dir']
    for part in kwargs['parts']:
        stl_path = os.path.join(dir, f'{part}.stl')
        if not os.path.exists(stl_path):
            # create stl files if possible
            if 'ldraw' in kwargs:
                dat_path = os.path.join(kwargs['ldraw'], f'{part}.dat')
                os.system(f'LDView64 {dat_path} -ExportFile={stl_path}')
            # remove part from list if part generation not available
            else:
                kwargs['parts'].remove(part)


    # TODO: load .blend files
    #   - maybe it will be run by "blender <scene>.blend -b --python generate.py -- <config>.cfg"

    # set render parameters
    render = bpy.context.scene.render
    render.engine = kwargs['engine']
    render.resolution_x, render.resolution_y = tuple(map(int, kwargs['resolution'].split('x')))

    # place and configure camera
    cam = bpy.context.scene.camera
    cam.location = (0, -30, 30)
    cam.rotation_euler = (math.radians(40), 0, 0)
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(60)

    # attach light source to camera
    light = bpy.data.objects['Light']
    bpy.context.view_layer.objects.active = light
    bpy.ops.object.constraint_add(type='COPY_LOCATION')
    bpy.context.object.constraints["Copy Location"].target = cam
    bpy.data.lights[0].energy = 100000

    # add background to render
    bpy.ops.preferences.addon_enable(module='io_import_images_as_planes')
    bg = add_background(random.choice(os.listdir('scenes')), kwargs)

    # generate dataset
    size = max(int(kwargs['size']), len(kwargs['split']))
    for fnum, fdr in enumerate(['train', 'val', 'test']):
        subsize = int(size * kwargs['split'][fnum])
        for _ in range(subsize):
            # clear legos and probabilistically remove the background
            keep = [bg if random.random() < 0.8 else None]
            remove_objects(['MESH'], keep=keep)
            if keep[0] is None:
                bg = add_background(random.choice(os.listdir('scenes')), kwargs)

            # add parts to scene
            num_parts = random.randint(0, int(kwargs['capacity']))
            parts = []
            for _ in range(num_parts):
                # get random part id
                part = random.choice(kwargs["parts"])

                # add masked lego object to scene
                # TODO: add distractors?
                lego = add_lego(part, kwargs)

                parts.append(kwargs['cnums'].index(part))

            # simulate gravity
            if kwargs['gravity'] == 'on':
                for f in range(100):
                    bpy.context.scene.frame_set(f)

            # render and save image and masks
            # TODO: better handling of kwargs
            box = kwargs['lego_area'] if 'lego_area' in kwargs else (0,) * 4
            popped = 0
            for idx, obj in enumerate(bpy.data.objects):
                if 'inst_id' not in obj:
                    continue
                loc = obj.matrix_world.translation
                if loc[0] < box[0] or loc[1] < box[1] or loc[0] > box[2] or loc[1] > box[3]:
                    bpy.data.objects.remove(obj)
                    parts.pop(idx - popped)
                    popped += 1
            save_to_set(parts, fdr, kwargs)


if __name__ == '__main__':
    argument_list = [
        ConfigArgument(),
        SingleArgument(name='engine', flag='--engine', shortcut='-e', default='CYCLES', values=[
            'BLENDER_EEVEE',
            'CYCLES'
        ]),
        ResolutionArgument(),
        SingleArgument(name='dataset', flag='--dataset', default='data'),
        SingleArgument(name='size', flag='--size', shortcut='-s', default='1000'),
        SingleArgument(name='gravity', flag='--gravity', shortcut='-g', default='on', values=['on', 'off']),
        SingleArgument(name='stl_dir', flag='--stl-dir', default='stls'),
        MultiArgument(name='parts', flag='--parts', shortcut='-p', default=[]),
        SingleArgument(name='capacity', flag='--cap', default='1'),
        SetSplitArgument()
    ]
    main_args = parse_args(argument_list, sys.argv)
    if 'exit' not in main_args:
        main(main_args)
