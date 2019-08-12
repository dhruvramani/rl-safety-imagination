import os
import numpy as np
from safe_grid_gym.envs.gridworlds_env import GridworldEnv

env = GridworldEnv("conveyor_belt")

ob_space = env.observation_space.shape
nc, nw, nh = ob_space

done = False
state = env.reset()
path = [3, 1, 1, 1]
for a in path:
    print(state)
    action = env.action_space.sample()
    state, reward, done, _ = env.step([a])
    print("Reward : ", reward)
