import gym
import env
from tqdm import tqdm
from wrappers import Discretization, VectorObsWrapper, obs_wrapper

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
    env = Discretization(env)
    env = obs_wrapper(env)
    env = VectorObsWrapper(env)
    print(f'action space: {env.action_space}')
    print(f'observation space: {env.observation_space}')
    print(env)
    print(f'current task: {env.spec._kwargs["env_spec"].task_monitor.current_task}')
    obs = env.reset()
    done = False
    tq = tqdm()
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        tq.update(1)
    tq.close()
    print(f'total reward {total_reward}')
