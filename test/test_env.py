
import minerl_patched
import iglu
import gym

from tqdm import tqdm
from time import time

if __name__ == '__main__':
    env = gym.make('IGLUSilentBuilder-v0', max_steps=1000)
    print(f'action space: {env.action_space}')
    print(f'observation space: {env.observation_space}')
    print(f'current task: {env.tasks.current}')
    obs = env.reset()
    done = False
    tq = tqdm(disable=False)
    total_reward = 0
    step = 1
    global_step = 0
    t = time()
    for i in range(20):
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            tq.update(1)
            step += 1
            global_step += 1
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
