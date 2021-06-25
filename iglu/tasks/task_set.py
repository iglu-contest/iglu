import os
import re
import json
import shutil
import sys
import pickle
import uuid
from zipfile import ZipFile
import pandas as pd
from collections import defaultdict

import numpy as np

from .load import download_file_from_google_drive
from .task import Task

from ..const import block2id, id2block, block_map, \
                BUILD_ZONE_SIZE, \
                BUILD_ZONE_SIZE_X, \
                BUILD_ZONE_SIZE_Z

DATA_PREFIX = os.path.join(os.environ['HOME'], '.iglu', 'data')


class TaskSet:
    ALL = {}
    def __init__(self, preset='simplest', task_id=None, update_task_dict=False):
        self._load_data(
            force_download=os.environ.get('IGLU_FORCE_DOWNLOAD', '0') == '1',
            update_task_dict=update_task_dict)
        self.tasks = self._parse_data()
        self.preset = {}
        task_set = None
        if isinstance(preset, list):
            task_set = preset
        elif preset == 'simplest':
            task_set = SIMPLEST_TASKS
        elif preset == 'one_task':
            task_set = [task_id]
        else:
            raise ValueError('Incorrect preset name!')
        self.task_ids = task_set
        self.current = None
        for task_id in task_set:
            if len(self.tasks[task_id]) == 0:
                continue
            task_path = os.path.join(DATA_PREFIX, self.tasks[task_id][0][0], 'logs', self.tasks[task_id][0][1])
            task = Task(*self._parse_task(task_path, task_id, update_task_dict=update_task_dict))
            self.preset[task_id] = task
        
    def sample(self):
        sample = np.random.choice(len(self.task_ids))
        self.current = self.preset[self.task_ids[sample]]
        return self.current

    def set_task(self, task_id):
        self.current = self.preset[task_id]
        return self.current

    def _load_data(self, force_download=False, update_task_dict=False):
        if update_task_dict:
            path = sys.modules[__name__].__file__
            path_dir, _ = os.path.split(path)
            tasks = pd.read_csv(os.path.join(path_dir, 'task_names.txt'), sep='\t', names=['task_id', 'name'])
            TaskSet.ALL = dict(tasks.to_records(index=False))
        if not os.path.exists(DATA_PREFIX):
            os.makedirs(DATA_PREFIX, exist_ok=True)
        path = os.path.join(DATA_PREFIX, 'data.zip')
        done = len(list(filter(lambda x: x.startswith('data-'), os.listdir(DATA_PREFIX)))) == 16
        if done and not force_download:
            return
        if force_download:
            for dir_ in os.listdir(DATA_PREFIX):
                if dir_.startswith('data-'):
                    shutil.rmtree(os.path.join(DATA_PREFIX, dir_), ignore_errors=True)
        if not os.path.exists(path) or force_download:
            download_file_from_google_drive(
                id='1_L94tQXhwAmQJIvYO0dd9eED60i9yw8C',
                destination=path,
                data_prefix=DATA_PREFIX
            )
            with ZipFile(path) as zfile:
                zfile.extractall(DATA_PREFIX)
        dir_name = 'The Minecraft Dialogue Corpus -- no screenshots'
        dirs = [dir_ for dir_ in os.listdir(os.path.join(DATA_PREFIX, dir_name)) if dir_.startswith('data-')]
        for dir_ in dirs:
            shutil.move(
                os.path.join(DATA_PREFIX, dir_name, dir_), 
                os.path.join(DATA_PREFIX, dir_),
            )
        for entry in os.listdir(DATA_PREFIX):
            if entry not in dirs:
                candidate = os.path.join(DATA_PREFIX, entry)
                if os.path.isfile(candidate):
                    os.remove(candidate)
                else:
                    shutil.rmtree(candidate, ignore_errors=True)
        
    def _parse_data(self):
        tasks = defaultdict(list)
        for folder in os.listdir(DATA_PREFIX):
            if folder.startswith('data-'):
                path = os.path.join(DATA_PREFIX, folder)
                with open(os.path.join(path, 'dialogue.txt'), 'r') as f:
                    for task_id in filter(lambda x: x.startswith('B'), f.readlines()):
                        tasks[re.search(r'C\d+', task_id).group()].append((folder, task_id.strip()))
        return tasks

    def _parse_task(self, path, task_id, update_task_dict=False):
        if not os.path.exists(path):
            # try to unzip logs.zip
            path_prefix, top = path, ''
            while top != 'logs':
                path_prefix, top = os.path.split(path_prefix)
            with ZipFile(os.path.join(path_prefix, 'logs.zip')) as zfile:
                zfile.extractall(path_prefix)
        with open(os.path.join(path, 'postprocessed-observations.json'), 'r') as f:
            data = json.load(f)
        data = data['WorldStates'][-1]
        chat = '\n'.join(data['ChatHistory'])
        target_grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int32)
        for block in data['BlocksInGrid']:
            coord = block['AbsoluteCoordinates']
            target_grid[
                coord['Y'] - 1,
                coord['X'] + 5,
                coord['Z'] + 5
            ] = block2id[block_map[block['Type']]]
        if update_task_dict:
            total_blocks = len(data['BlocksInGrid'])
            colors = len(np.unique([b['Type'] for b in data['BlocksInGrid']]))
            TaskSet.ALL[task_id] = f'{TaskSet.ALL[task_id]} ({total_blocks} blocks, {colors} colors)'
        return chat, target_grid

    def __repr__(self):
        tasks = ", ".join(f'"{t}"' for t in self.task_ids)
        return f'TaskSet({tasks})'
    
    @staticmethod
    def subset(task_set):
        return {k: v for k, v in TaskSet.ALL.items() if k in task_set}


class RandomTasks(TaskSet):
    """ 
    TaskSet that consists of number of randomly generated tasks

    Args:
        max_blocks (``int``): The maximal number of blocks in each task. Defaults to 3.
        height_levels (``int``): How many height levels random blocks can occupy. Defaults to 1.
        allow_float (``bool``): Whether to allow blocks to have the air below. Defaults to False.
        max_dist (``int``): Maximum (Chebyshev) distance between two blocks. Defaults to 2.
        num_colors (``int``): Maximum number of unique colors. Defaults to 1.
        max_cache (``int``): If zero, each `.sample_task` will generate new task. Otherwise, the number of random tasks to cache. Defaults to 0.

    """
    def __init__(
        self, max_blocks=4,
        height_levels=1, allow_float=False, max_dist=2, 
        num_colors=1, max_cache=0, 
    ): 
        self.height_levels = height_levels
        self.max_blocks = max_blocks
        self.allow_float = allow_float
        self.max_dist = max_dist
        self.num_colors = num_colors
        self.max_cache = max_cache
        self.preset = {}
        self.current = None
        for _ in range(self.max_cache):
            uid = str(uuid.uuid1())[:8]
            self.preset[uid] = self.sample_task()
        self.sample()

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump({uid: t.target_grid for uid, t in self.preset.items()}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            grids = pickle.load(f)
        self.preset = {uid: Task('', g) for uid, g in grids.items()}

    def __repr__(self):
        hps = dict(
            max_blocks=self.max_blocks,
            height_levels=self.height_levels,
            allow_float=self.allow_float,
            max_dist=self.max_dist,
            num_colors=self.num_colors,
            max_cache=self.max_cache,
        )
        hp_str = ', '.join(f'{k}={v}' for k, v in hps.items())
        return f'RandomTasks({hp_str})'

    def sample(self):
        if self.max_cache > 0:
            sample = np.random.choice(list(self.preset.keys()))
            self.current = self.preset[sample]
            self.current_id = sample
            return self.current
        else:
            self.current = self.sample_task()
            return self.current

    def set_task(self, task_id):
        self.current = self.preset[task_id]
        return self.current

    def sample_task(self):
        chat = ''
        target_grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int32)
        for height in range(self.height_levels):
            shape = target_grid[height].shape
            block_x = np.random.choice(BUILD_ZONE_SIZE_X)
            block_z = np.random.choice(BUILD_ZONE_SIZE_Z)
            color = np.random.choice(self.num_colors) + 1
            target_grid[height, block_x, block_z] = color
            for _ in range(self.max_blocks - 1):
                block_delta_x, block_delta_z = 0, 0
                while block_delta_x == 0 and block_delta_z == 0 \
                        or block_x + block_delta_x >= BUILD_ZONE_SIZE_X \
                        or block_z + block_delta_z >= BUILD_ZONE_SIZE_Z \
                        or block_x + block_delta_x < 0 \
                        or block_z + block_delta_z < 0 \
                        or target_grid[height, block_x + block_delta_x, block_z + block_delta_z] != 0:
                    block_delta_x = np.random.choice(2 * self.max_dist + 1) - self.max_dist 
                    block_delta_z = np.random.choice(2 * self.max_dist + 1) - self.max_dist
                    color = np.random.choice(self.num_colors) + 1
                target_grid[height, block_x + block_delta_x, block_z + block_delta_z] = color

        return Task(chat, target_grid)


SIMPLEST_TASKS = TaskSet.subset(['C3', 'C8', 'C12', 'C14', 'C32', 'C17'])