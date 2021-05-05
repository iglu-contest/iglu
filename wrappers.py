import gym
from minerl.herobraine.hero.handlers.agent.actions import camera
import numpy as np
import wandb
from copy import deepcopy


class ImageOnlyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space['image']

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation['image'], reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation['image']


class BoundSteps(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.step_n = 0

    def reset(self):
        self.step_n = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_n += 1
        if self.step_n == self.max_steps:
            done = True
        return obs, reward, done, info


class ActionReductionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self._action_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 13, 18]
        self._action_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        # self._action_mapping = [1,2,3,4,]
        print('new actions:')
        print([env.action_names[0][i] for i in self._action_mapping])
        self.action_space = gym.spaces.Discrete(len(self._action_mapping))

    def action(self, action):
        return self._action_mapping[action]

    def reverse_action(self, action):
        pass


class Discretization(gym.ActionWrapper):
    def __init__(self, env, ):
        super().__init__(env)
        camera_delta = 5
        binary = ['attack', 'use', 'forward', 'back', 'left', 'right', 'jump']
        discretes = []
        for op in binary:
            dummy = env.action_space.no_op()
            dummy[op] = 1
            discretes.append(dummy)
        camera_x = env.action_space.no_op()
        camera_x['camera'][0] = camera_delta
        discretes.append(camera_x)
        camera_x = env.action_space.no_op()
        camera_x['camera'][0] = -camera_delta
        discretes.append(camera_x)
        camera_y = env.action_space.no_op()
        camera_y['camera'][1] = camera_delta
        discretes.append(camera_y)
        camera_y = env.action_space.no_op()
        camera_y['camera'][1] = -camera_delta
        discretes.append(camera_y)
        for i in range(6):
            dummy = env.action_space.no_op()
            dummy['hotbar'] = i + 1
            discretes.append(dummy)
        discretes.append(env.action_space.no_op())
        self.discretes = discretes
        self.action_space = gym.spaces.Discrete(len(discretes))

    def action(self, action):
        return self.discretes[action]

    def reverse_action(self, action):
        pass

import cv2
import gym

class obs_wrapper(gym.Wrapper):
    def __init__(self, env, filename='./mine_videos/vid.mp4', fps=30):
        super().__init__(env)
        self.filename = filename
        self.fps = fps
        self.frames = []
        self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (500, 500))
        # self.observation_space = env.observation_space['pov']

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.out.write(obs['pov'][..., ::-1])
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs

class VectorObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-360, 360, (1094,))

    def reset(self):
        return self.change_obs(self.env.reset())

    @staticmethod
    def change_obs(obs):
        return np.concatenate([obs['agentPos'], obs['grid'].flatten()])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.change_obs(obs), reward, done, info


class RewardMonitor(gym.Wrapper):
    def __init__(self, env, wb=True):
        super().__init__(env)
        self._episode_reward = None
        self.total_steps = 0
        if wb:
            self.wandb = wandb.init(project='PPO-nips-comp-baseline', tags=['vector'])
        else:
            self.wandb = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._episode_reward += reward
        self.total_steps += 1
        return observation, reward, done, info

    def reset(self, **kwargs):
        if self._episode_reward is not None:
            if self.wandb:
                self.wandb.log({"total_steps": self.total_steps, "reward": self._episode_reward})
        self._episode_reward = 0
        return self.env.reset(**kwargs)
