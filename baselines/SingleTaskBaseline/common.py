from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.logger import Logger
from typing import Optional, Any, Union, Tuple


class WbLogger(Logger):
    def __init__(self, wb):
        super().__init__(None, [])
        self.wb = wb
        self.steps = 0

    def dump(self, step: int = 0) -> None:
        pass

    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        if key == 'time/total_timesteps':
            self.steps = value
        else:
            self.wb.log({key: value, 'train/steps': self.steps})


class WbRewardCallback(BaseCallback):
    def __init__(self, wb=None, prefix='train'):
        super().__init__()
        self.rewards = None
        self.wb = wb
        self.mean_reward = 0
        self.prefix = prefix

    def _on_step(self):

        done_array = np.array(
            self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        reward_array = np.array(
            self.locals.get("rewards") if self.locals.get("reward") is not None else self.locals.get("rewards"))
        if self.rewards is None:
            self.rewards = np.zeros(done_array.shape)
        self.rewards += reward_array
        if done_array.sum() > 0:
            self.wb.log({f"{self.prefix}/steps": self.num_timesteps,
                         f"{self.prefix}/reward": np.sum(self.rewards * done_array) / done_array.sum()})
            self.rewards *= 1 - done_array