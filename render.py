import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence


BLENDER_INSTALLATION_PATH = '/home/li325/Downloads'

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'
# BLENDER_PATH = f'/usr/bin/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, output_dir, num_views=10):
    filename = file_path.split('/')[-1].split('.')[0]
    print('start', filename)
    output_folder = os.path.join(output_dir, 'renders', filename)

    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    args = [
        BLENDER_PATH, '-b', '-P', 'blender_script/render.py',
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--save_depth',
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    # call(args, stdout=DEVNULL, stderr=DEVNULL)
    call(args)
    print('finished', filename)

    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'rendered': True}


if __name__ == '__main__':
    # this is only for validating, and is not meant for batched rendering.
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='renderings',
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')

    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)

    # install blender
    # print('Checking blender...', flush=True)
    # _install_blender()

    # get file list
    objaverse_root = '/home/li325/project/objaverse_ws/objervse_sketchfab/old/raw.old/hf-objaverse-v1/glbs'
    subset = '000-036'
    all_obj_list = os.listdir(os.path.join(objaverse_root, subset))

    selected = all_obj_list[:1]

    for object_path in selected:
        object_path = os.path.join(objaverse_root, subset, object_path)
        print(object_path)
        _render(object_path, opt.output_dir, opt.num_views)
