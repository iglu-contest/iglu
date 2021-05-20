import gym
# import cv2
import iglu
from iglu.tasks import RandomTasks, TaskSet
import sys
from tqdm import tqdm
# from wrappers import Discretization, VectorObsWrapper, obs_wrapper
import logging 

# logging.basicConfig(stream=sys.stdout)
# logging.getLogger('env.handlers').setLevel(logging.DEBUG)
# logging.getLogger('wrappers').setLevel(logging.DEBUG)

def build_env(config):
    env = gym.make('IGLUSilentBuilder-v0')
    env = Discretization(env)
    env = VectorObsWrapper(env)
    print(f'action space: {env.action_space}')
    print(f'observation space: {env.observation_space}')
    print(env)
    print(f'current task: {env.spec._kwargs["env_spec"].task_monitor.current_task}')
    return env

if __name__ == '__main__':
    env = gym.make('IGLUSilentBuilder-v0')
    # env = obs_wrapper(env)
    # env = Discretization(env)
    # env = VectorObsWrapper(env)
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
    while not done:
        action = env.action_space.sample()
        # action = 0
        obs, reward, done, info = env.step(action)
        
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
