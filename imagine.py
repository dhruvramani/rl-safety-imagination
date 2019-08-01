import os
import sys
import gym
import time
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils import SubprocVecEnv
from a2c import get_actor_critic, CnnPolicy
from env_model import make_env, create_env_model
from safe_grid_gym.envs.gridworlds_env import GridworldEnv
from discretize_env import pix_to_target, target_to_pix, rewards_to_target, _NUM_PIXELS, sokoban_rewards

np.set_printoptions(threshold=sys.maxsize)

# Source : https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta = 1.0, axis = None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p

def convert_target_to_real(batch_size, nw, nh, nc, imagined_state, imagined_reward):
    imagined_state = softmax(imagined_state, axis=1)
    imagined_state = np.argmax(imagined_state, axis=1)
    imagined_state = target_to_pix(imagined_state)
    imagined_state = imagined_state.reshape((batch_size, nc, nw, nh))

    imagined_reward = softmax(imagined_reward, axis=1)
    imagined_reward = np.argmax(imagined_reward, axis=1)

    return imagined_state, imagined_reward


class ImaginationCore(object):
    def __init__(self, num_actions, num_rewards, ob_space, env_model, num_rollouts=1):

        self.num_rollouts = num_rollouts
        self.num_actions  = num_actions
        self.num_rewards  = num_rewards
        self.ob_space     = ob_space
        self.env_model    = env_model


    def imagine(self, state, sess, act):
        nc, nw, nh = self.ob_space

        batch_size = state.shape[0]

        state = np.tile(state, [self.num_actions, 1, 1, 1, 1])
        state = state.reshape(-1, nw, nh, nc)

        action = act
        rollout_batch_size = batch_size * self.num_actions

        rollout_states = []
        rollout_rewards = []

        for step in range(self.num_rollouts):
            state = state.reshape(-1, nw, nh, nc)

            onehot_action = np.zeros((rollout_batch_size, self.num_actions, nw, nh))
            onehot_action[range(rollout_batch_size), action] = 1
            onehot_action = np.transpose(onehot_action, (0, 2, 3, 1))

            imagined_state, imagined_reward = sess.run(
                    [self.env_model.imag_state, self.env_model.imag_reward],
                    feed_dict={
                        self.env_model.input_states: state,
                        self.env_model.input_actions: onehot_action,
                })

            imagined_state, imagined_reward = convert_target_to_real(rollout_batch_size, nw, nh, nc, imagined_state, imagined_reward)

            onehot_reward = np.zeros((rollout_batch_size, self.num_rewards))
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            rollout_states.append(imagined_state)
            rollout_rewards.append(onehot_reward)

            state = imagined_state
            state = state.reshape(-1, nw, nh, nc)
            #action, _, _ = self.actor_critic.act(state)

        return rollout_states, rollout_rewards
