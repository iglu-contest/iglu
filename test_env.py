import iglu
import minerl
import pickle
import json
import gym
import numpy as np
from time import time
from collections import defaultdict
from iglu.tasks import RandomTasks, TaskSet
import sys
from tqdm import tqdm
import logging

if __name__ == '__main__':
    env = gym.make('IGLUSilentBuilderVisual-v0', action_space='discrete')
    print(f'action space: {env.action_space}')
    print(f'observation space: {env.observation_space}')
    print(f'env tasks: {env.tasks}')
    obs = env.reset()
    done = False
    tq = tqdm(disable=False)
    total_reward = 0
    step = 1
    global_step = 0
    t = time()
    action = None
    for i in range(1):
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            total_reward += reward
            tq.update(1)
            step += 1
            global_step += 1
        robs = env.reset()
        done = False
        step = 1
        tq.close()
        print(f'episode reward: {total_reward}')
        total_reward = 0
        tq = tqdm(disable=False)
    delta = time() - t
    print(f'total time: {delta:.4f}, fps: {global_step / delta:.4f}')
    tq.close()
    env.close()
