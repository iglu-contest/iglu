import cv2
import gym

class obs_wrapper(gym.Wrapper):
    def __init__(self, env, filename='./mine_videos/vid.mp4', fps=30):
        super().__init__(env)
        self.filename = filename
        self.fps = fps
        self.frames = []
        self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 
                                   fps, (64, 64))
        self.observation_space = env.observation_space['pov']

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.out.write(obs['pov'][..., ::-1]) 
        return obs['pov'], reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs['pov']


class RewardFromInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = info['reward']
        del info['reward']
        return obs, reward, done, info