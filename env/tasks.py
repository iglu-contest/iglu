import os
import json
from zipfile import ZipFile

import numpy as np

from .const import BUILD_ZONE_SIZE_X, \
                   BUILD_ZONE_SIZE_Y, \
                   BUILD_ZONE_SIZE_Z, \
                   GROUND_LEVEL, \
                   block_map, block2id

tasks = {
    'blue3L': '/tmp/data/data-4-13/logs/B34-A31-C3-1523649983538/',
    'orange3L_horiz': '/tmp/data/data-4-16/logs/B36-A35-C17-1523916463263/',
    'blue5L': '/tmp/data/data-4-13/logs/B35-A44-C32-1523650068421/',
    'table10': '/tmp/data/data-4-10/logs/B15-A38-C8-1523398549691/',
    'heart': '/tmp/data/data-4-10/logs/B15-A38-C11-1523400016265/'
}

class Task:
    def load(self, path):
        if not os.path.exists(path):
            import pdb
            pdb.set_trace()
            # try to unzip logs.zip
            path_prefix, top = path, ''
            while top != 'logs':
                path_prefix, top = os.path.split(path_prefix)
            with ZipFile(os.path.join(path_prefix, 'logs.zip')) as zfile:
                zfile.extractall(path_prefix)
        with open(os.path.join(path, 'postprocessed-observations.json'), 'r') as f:
            data = json.load(f)
        data = data['WorldStates'][-1]
        self.chat = '\n'.join(data['ChatHistory'])
        self.target_grid = np.zeros((9, 11, 11), dtype=np.int32)
        for block in data['BlocksInGrid']:
            coord = block['AbsoluteCoordinates']
            self.target_grid[
                coord['Y'] - 1,
                coord['X'] + 5,
                coord['Z'] + 5
            ] = block2id[block_map[block['Type']]]
        self.admissible = [[] for _ in range(4)]
        self.target_size = (self.target_grid != 0).sum().item()
        self.target_grids = [self.target_grid]
        # fill self.target_grids with four rotations of the original grid around vertical axis
        for _ in range(3):
            self.target_grids.append(np.zeros(self.target_grid.shape, dtype=np.int32))
            for x in range(BUILD_ZONE_SIZE_X):
                for z in range(BUILD_ZONE_SIZE_Z):
                    self.target_grids[-1][:, z, BUILD_ZONE_SIZE_X - x - 1] \
                        = self.target_grids[-2][:, x, z]
        # (dx, dz) is admissible iff the translation of target grid by (dx, dz) preserve
        # target structure within original (unshifted) target grid
        for i in range(4):
            for dx in range(-BUILD_ZONE_SIZE_X + 1, BUILD_ZONE_SIZE_X):
                for dz in range(-BUILD_ZONE_SIZE_Z + 1, BUILD_ZONE_SIZE_Z):
                    sls_target = self.target_grids[i][:, max(dx, 0):BUILD_ZONE_SIZE_X + min(dx, 0),
                                                         max(dz, 0):BUILD_ZONE_SIZE_Z + min(dz, 0):]
                    if (sls_target != 0).sum().item() == self.target_size:
                        self.admissible[i].append((dx, dz))

    def maximal_intersection(self, grid):
        max_int = 0
        for i, admissible in enumerate(self.admissible):
            for dx, dz in admissible:
                x_sls = slice(max(dx, 0), BUILD_ZONE_SIZE_X + min(dx, 0))
                z_sls = slice(max(dz, 0), BUILD_ZONE_SIZE_Z + min(dz, 0))
                sls_target = self.target_grids[i][:, x_sls, z_sls]

                x_sls = slice(max(-dx, 0), BUILD_ZONE_SIZE_X + min(-dx, 0))
                z_sls = slice(max(-dz, 0), BUILD_ZONE_SIZE_Z + min(-dz, 0))
                sls_grid = grid[:, x_sls, z_sls]
                intersection = ((sls_target == sls_grid) & (sls_target != 0)).sum().item()
                if intersection > max_int:
                    max_int = intersection
        return max_int
