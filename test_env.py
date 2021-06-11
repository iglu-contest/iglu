import sys
sys.path.insert(0, '/home/artem/experiments/rl/minerl/iglu_minerl/')
import minerl
import pickle
import gym
# import cv2
import numpy as np
from time import time
from collections import defaultdict
import iglu
from iglu.env import FakeResetWrapper
from iglu.tasks import RandomTasks, TaskSet
import sys
from tqdm import tqdm
from wrappers import Discretization, VectorObsWrapper, obs_wrapper
import logging 

if __name__ == '__main__':
    env = gym.make('IGLUSilentBuilder-v0', max_steps=500)
    env = obs_wrapper(env)
    # env = Discretization(env)
    # env = VectorObservationWrapper(env)
    print(f'action space: {env.action_space}')
    print(f'observation space: {env.observation_space}')
    print(f'env tasks: {env.tasks}')
    print(env)
    print(f'current task: {env.tasks.current}')
    env.update_taskset(TaskSet(preset='one_task', task_id='C32'))
    # print('making tasks random...')
    # env.update_taskset(RandomTasks())
    # print(f'new task set: {env.tasks}')
    obs = env.reset()
    done = False
    tq = tqdm(disable=False)
    total_reward = 0
    step = 1
    global_step = 0
    with open('converted.pkl', 'rb') as f:
        act_list = pickle.load(f)
    disc_actions = env.spec._kwargs['env_spec'].actionables[0]
    t = time()
    for i in range(20):
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            tq.update(1)
            step += 1
            global_step += 1
        # env.should_reset
        env.reset()
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
    print(f'total reward {total_reward}')
