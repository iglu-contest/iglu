import wandb
from gym import ObservationWrapper
import gym
import numpy as np
from iglu.tasks import TaskSet
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from common import WbRewardCallback, WbLogger
from wrappers import IgluActionWrapper
from iglu.tasks.task_set import TaskSet

class PovOnlyWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)

    def observation(self, observation):
        return observation['pov']


def make_pov_iglu(task='C3'):
    def make():
        env = gym.make('IGLUSilentBuilder-v0', max_steps=1000, )
        env.update_taskset(TaskSet(preset=[task]))
        env = PovOnlyWrapper(env)
        env = IgluActionWrapper(env)

        return env

    return make


def main(num_proc=1):
    task = 'C17'
    wb = wandb.init(project='IGLU-SingleTaskBaseline', config={'task': task})
    if num_proc > 1:
        env = SubprocVecEnv([make_pov_iglu() for _ in range(num_proc)])
    else:
        env = make_pov_iglu()()
    agent = PPO('CnnPolicy', env, verbose=True)
    
    agent.set_logger(WbLogger(wb))
    agent.learn(5000000, 
            callback=[WbRewardCallback(wb=wb)]
            )
    agent.save

if __name__ == '__main__':
    main()

