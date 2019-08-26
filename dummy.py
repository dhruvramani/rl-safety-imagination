import os
import numpy as np
from safe_grid_gym.envs.gridworlds_env import GridworldEnv

env = GridworldEnv("conveyor_belt")

ob_space = env.observation_space.shape
nc, nw, nh = ob_space

done = False
state = env.reset()
path = [1, 1, 2, 1, 1, 3, 0, 0, 0]
for a in path:
    print(state)
    action = env.action_space.sample()
    state, reward, done, _ = env.step([a])
    print("Reward : ", reward)
