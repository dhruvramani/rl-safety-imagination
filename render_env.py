import os
import time
import curses
import numpy as np
import tensorflow as tf
from env_model import make_env, create_env_model
from utils import SubprocVecEnv
from discretize_env import pix_to_target, rewards_to_target, _NUM_PIXELS, sokoban_rewards
from a2c import get_actor_critic, CnnPolicy
from imagine import convert_target_to_real
from safe_grid_gym.envs.gridworlds_env import GridworldEnv

nenvs = 16
nsteps = 5
envs = [make_env() for i in range(nenvs)]
envs = SubprocVecEnv(envs)

ob_space = envs.observation_space.shape
ac_space = envs.action_space
num_actions = envs.action_space.n

env = GridworldEnv("conveyor_belt")

done = False
states = env.reset()
num_actions = ac_space.n
nc, nw, nh = ob_space
print('Observation space ', ob_space)
print('Number of actions ', num_actions)
steps = 0

with tf.Session() as sess:
    with tf.variable_scope('actor'):
        actor_critic = get_actor_critic(sess, nenvs, nsteps, ob_space,
                ac_space, CnnPolicy, should_summary=False)
    actor_critic.load('a2c_weights/3.0/a2c3.0_3100.ckpt')
    
    with tf.variable_scope('env_model'): 
        env_model = create_env_model(ob_space, num_actions,_NUM_PIXELS,
                    len(sokoban_rewards), should_summary=False)

    save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
    loader = tf.train.Saver(var_list=save_vars)
    loader.restore(sess, 'weights/env_model.ckpt')
    
    while not done and steps < 20:
        steps += 1
        actions, _, _ = actor_critic.act(np.expand_dims(states, axis=3))

        onehot_actions = np.zeros((1, num_actions, nw, nh))
        onehot_actions[range(1), actions] = 1
        # Change so actions are the 'depth of the image' as tf expects
        onehot_actions = onehot_actions.transpose(0, 2, 3, 1)

        '''
        s, r = sess.run([env_model.imag_state, 
                                        env_model.imag_reward], 
                                       feed_dict={
                env_model.input_states: np.expand_dims(states, axis=3),
                env_model.input_actions: onehot_actions
            })
        
        s, r = convert_target_to_real(1, nw, nh, nc, s, r)
        '''
        states, reward, done, _ = env.step(actions[0])
        env.render()
        # NOTE : render screws up if reward isnt proper
        '''
        env.render("human", states[0, :, :], reward)
        #env.render("human", s[0, 0, :, :], sokoban_rewards[r[0]])
        time.sleep(0.2)
        '''
env.close()
