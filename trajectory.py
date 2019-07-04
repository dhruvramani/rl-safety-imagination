import os
import sys
import gym
import time
import logging
import numpy as np
import tensorflow as tf
from safe_grid_gym.envs.gridworlds_env import GridworldEnv
from utils import SubprocVecEnv
from tqdm import tqdm

from env_model import make_env, create_env_model
from a2c import get_actor_critic, CnnPolicy
from discretize_env import pix_to_target, target_to_pix, rewards_to_target, _NUM_PIXELS, sokoban_rewards

np.set_printoptions(threshold=sys.maxsize)

# Hyperparameter of how far ahead in the future the agent "imagines"
# Currently this is specifying one frame in the future.
NUM_ROLLOUTS = 1

# Hidden size in RNN imagination encoder.
HIDDEN_SIZE = 256

N_ENVS = 16
N_STEPS = 5

# Replace this with the name of the weights you want to load to train I2A
A2C_MODEL_PATH = 'weights/a2c_1800.ckpt'
ENV_MODEL_PATH = 'weights/env_model.ckpt'

# Softmax function for numpy taken from
# https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
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

# So the model is not loaded twice.
g_actor_critic = None
def get_cache_loaded_a2c(sess, nenvs, nsteps, ob_space, ac_space):
    global g_actor_critic
    if g_actor_critic is None:
        with tf.variable_scope('actor'):
            g_actor_critic = get_actor_critic(sess, nenvs, nsteps, ob_space,
                    ac_space, CnnPolicy, should_summary=False)
        g_actor_critic.load(A2C_MODEL_PATH)

        print('Actor restored!')
    return g_actor_critic


# So the model is not loaded twice.
g_env_model = None
def get_cache_loaded_env_model(sess, ob_space, num_actions):
    global g_env_model
    if g_env_model is None:
        with tf.variable_scope('env_model'):
            g_env_model = create_env_model(ob_space, num_actions, _NUM_PIXELS,
                    len(sokoban_rewards), should_summary=False)

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
        loader = tf.train.Saver(var_list=save_vars)
        loader.restore(sess, ENV_MODEL_PATH)

        print('Env model restored!')

    return g_env_model


"""
Used to generate rollouts of imagined states.
"""
class ImaginationCore(object):
    def __init__(self, num_rollouts, num_actions, num_rewards, ob_space, actor_critic, env_model):

        self.num_rollouts = num_rollouts
        self.num_actions  = num_actions
        self.num_rewards  = num_rewards
        self.ob_space     = ob_space
        self.actor_critic = actor_critic
        self.env_model    = env_model


    def imagine(self, state, sess):
        nc, nw, nh = self.ob_space

        batch_size = state.shape[0]

        state = np.tile(state, [self.num_actions, 1, 1, 1, 1])
        state = state.reshape(-1, nw, nh, nc)

        action = np.array([[[i] for i in range(self.num_actions)] for j in
            range(batch_size)])

        action = action.reshape((-1,))

        rollout_batch_size = batch_size * self.num_actions

        rollout_states = []
        rollout_rewards = []

        for step in range(self.num_rollouts):
            #state = np.squeeze(state, axis=1)
            #state = np.expand_dims(state, axis=3)
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
            action, _, _ = self.actor_critic.act(state)

        return rollout_states, rollout_rewards


def generate_trajectory(sess, state, ob_space, ac_space):
    num_actions = ac_space.n
    num_rewards = len(sokoban_rewards)

    actor_critic = get_cache_loaded_a2c(sess, N_ENVS, N_STEPS, ob_space, ac_space)
    env_model = get_cache_loaded_env_model(sess, ob_space, num_actions)

    imagination = ImaginationCore(NUM_ROLLOUTS, num_actions, num_rewards,
                ob_space, actor_critic, env_model)

    imagined_states, imagined_rewards = imagination.imagine(state, sess)
    return imagined_states, imagined_rewards

if __name__ == '__main__':
    envs = [make_env() for i in range(N_ENVS)]
    envs = SubprocVecEnv(envs)
    env = GridworldEnv("side_effects_sokoban")
    env.reset()

    ob_space = envs.observation_space.shape
    ac_space = envs.action_space
    
    obs = envs.reset()
    ob_np = np.copy(obs)
    ob_np = np.squeeze(ob_np, axis=1)
    ob_np = np.expand_dims(ob_np, axis=3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    imagined_states, imagined_rewards = generate_trajectory(sess, ob_np, ob_space, ac_space)

    for imagined_state in imagined_states:
        print(imagined_state)
        env.render("human", imagined_state[0, :, :], 0)
        time.sleep(0.2)
