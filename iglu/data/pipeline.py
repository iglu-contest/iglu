import pathlib
import pickle
import cv2
import re
import os
import numpy as np
import bz2
import pickle

from minerl_patched.data.util.constants import ACTIONABLE_KEY, HANDLER_TYPE_SEPERATOR, MONITOR_KEY, OBSERVABLE_KEY, REWARD_KEY
from minerl_patched.data.data_pipeline import DataPipeline as MineRLDataPipeline

from iglu.tasks.task_set import DATA_PREFIX
from iglu.const import BUILD_ZONE_SIZE


class IGLUDataPipeline(MineRLDataPipeline):
    def __init__(self, 
                 data_directory: os.path,
                 num_workers: int,
                 worker_batch_size: int,
                 min_size_to_dequeue: int,
                 random_seed=42):
        super().__init__(
            data_directory=data_directory,
            environment='IGLUSilentBuilder-v0',
            num_workers=num_workers, worker_batch_size=worker_batch_size,
            min_size_to_dequeue=min_size_to_dequeue,
            random_seed=random_seed
        )

    @classmethod
    def read_files(cls, file_dir):
        video_path = f'{file_dir}.mp4'
        numpy_path = f'{file_dir}.npz'
        action_path = f'{file_dir}_session.pkl'
        cap = cv2.VideoCapture(str(video_path))
        state = np.load(numpy_path, allow_pickle=True)
        state = {k: v for k, v in state.items()}
        with open(action_path, 'rb') as f:
            compressed_session = f.read()
        session = pickle.loads(bz2.decompress(compressed_session))
        dialogs = [d for d in session.dialogs if d is not None]
        a = re.compile(r'^A:')
        b = re.compile(r'^B:')
        dialogs = [a.sub('<Architect>', d) for d in dialogs]
        dialogs = [b.sub('<Builder>', d) for d in dialogs]
        dialog = '\n'.join(dialogs)
        meta = {
            'dialog': np.array([dialog for _ in range(len(state['reward']))], dtype=np.object),
            'target': [session.target  for _ in range(len(state['reward']))]
        }
        return cap, state, meta

    @classmethod
    def postprocess_batches(cls, batches):
        (current_observation, action, reward, next_observation, done), \
            rest = batches[:5], batches[5:]
        monitor, meta = None, None
        if len(rest) != 0:
            monitor = rest.pop(0)
        if len(rest) != 0:
            meta = rest.pop(0)
        for data in [current_observation, next_observation]:
            sparse_grids = data['grid'].tolist()
            grids = []
            for sparse_grid in sparse_grids:
                grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int)
                idx = [s[:3] for s in sparse_grid]
                vals = [s[3] for s in sparse_grid]
                if len(idx) != 0:
                    grid[tuple(zip(*idx))] = vals
                grids.append(grid)
            data['grid'] = np.stack(grids, axis=0)
            data['compass'] = {'angle': data['compass.angle']}
            del data['compass.angle']
        batches = [current_observation, action, reward, next_observation, done]
        if monitor is not None:
            batches.append(monitor)
        if meta is not None:
            batches.append(meta)
        return batches

    @classmethod
    def get_worker(cls):
        return iglu_job

    @staticmethod
    def _get_all_valid_recordings(path):
        path = pathlib.Path(path)
        sessions = path.glob('*-c*.mp4')
        sessions = [str(s)[:-len('.mp4')] for s in sessions]
        sessions = np.array(sessions)
        np.random.shuffle(sessions)
        sessions = sessions.tolist()
        # TODO: filter
        return sessions


def iglu_job(arg):
    return IGLUDataPipeline._load_data_pyfunc(*arg, include_metadata=True, include_monitor_data=True)
        