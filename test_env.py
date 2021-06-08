import sys
sys.path.insert(0, '/home/artem/experiments/rl/minerl/iglu_minerl/')
import minerl
import pickle
import gym
# import cv2
from collections import defaultdict
import iglu
from iglu.tasks import RandomTasks, TaskSet
import sys
from tqdm import tqdm
from wrappers import Discretization, VectorObsWrapper, obs_wrapper
import logging 

if __name__ == '__main__':
    env = gym.make('IGLUSilentBuilder-v0')
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
    tq = tqdm()
    total_reward = 0
    step = 0
    with open('converted.pkl', 'rb') as f:
        act_list = pickle.load(f)
    disc_actions = env.spec._kwargs['env_spec'].actionables[0]
    while not done:
        if step == len(act_list):
            break
        action = env.action_space.sample()
        nav_action = disc_actions.inverse_map[act_list[step]]
        action['navigation'] = nav_action
        # action = 0
        command = env.spec._kwargs['env_spec'].actionables[0].action_map[action['navigation']]
        prevObs = obs['agentPos'].copy()
        obs, reward, done, info = env.step(action)
        
        # pos = dict(zip(['x', 'y', 'z', 'p', 'yaw'], obs['agentPos']))
        # if 'jump' in command:
        #     print('---------------------')
        #     print(f'action: {command}')
        #     print(f'pos: {pos}')
        #     diff = dict(zip(['x', 'y', 'z', 'p', 'yaw'], obs['agentPos'] - prevObs))
        #     print(f'diff: {diff}')
        #     print('----------------------')
        # if step == 200:
        #     pov = obs['pov']
        #     cv2.imwrite('img.png', pov[..., ::-1])
        #     break
        total_reward += reward
        tq.update(1)
        step += 1
    tq.close()
    env.close()
    print(f'total reward {total_reward}')
