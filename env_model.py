import os
import copy
import tensorflow as tf
from safe_grid_gym.envs.gridworlds_env import GridworldEnv
from a2c import get_actor_critic, CnnPolicy
from utils import SubprocVecEnv
import numpy as np

from tqdm import tqdm
from discretize_env import pix_to_target, rewards_to_target, _NUM_PIXELS, conveyer_rewards

# How many iterations we are training the environment model for.
ENV_NAME = "conveyor_belt"
NUM_UPDATES = 50000
LOG_INTERVAL = 100
N_ENVS = 1
N_STEPS = 5
N_ACTIONS = 4

def make_env():
    def _thunk():
        env = GridworldEnv(ENV_NAME)
        return env
    return _thunk

envs = [make_env() for i in range(N_ENVS)]
envs = SubprocVecEnv(envs)
#envs = GridworldEnv("side_effects_sokoban")

ob_space = envs.observation_space.shape
nc, nw, nh = ob_space
ac_space = envs.action_space
num_actions = envs.action_space.n

def pool_inject(X, batch_size, depth, width, height):
    m = tf.layers.max_pooling2d(X, pool_size=(width, height), strides=(width, height))
    tiled = tf.tile(m, (1, width, height, 1))
    return tf.concat([tiled, X], axis=-1)

def basic_block(X, batch_size, depth, width, height, n1, n2, n3):
    with tf.variable_scope('pool_inject'):
        p = pool_inject(X, batch_size, depth, width, height)

    with tf.variable_scope('part_1_block'):
        # Padding was 6 here
        p_1_c1 = tf.layers.conv2d(p, n1, kernel_size=1,
                    padding='valid', activation=tf.nn.relu)
        p_1_c1 = tf.layers.dropout(p_1_c1, rate=0.5)

        # Padding was 5, 6
        p_1_c1 = tf.pad(p_1_c1, [[0,0],[1,1],[1,1],[0,0]])
        p_1_c2 = tf.layers.conv2d(p_1_c1, n1, kernel_size=3, strides=1,
                padding='valid', activation=tf.nn.relu)
        p_1_c2 = tf.layers.dropout(p_1_c2, rate=0.5)

    with tf.variable_scope('part_2_block'):
        p_2_c1 = tf.layers.conv2d(p, n2, kernel_size=1,
                activation=tf.nn.relu)
        p_2_c1 = tf.layers.dropout(p_2_c1, rate=0.5)
        p_2_c1 = tf.pad(p_2_c1, [[0,0],[1,1],[1,1],[0,0]])
        p_2_c2 = tf.layers.conv2d(p_2_c1, n2, kernel_size=3, strides=1,
                padding='valid', activation=tf.nn.relu)
        p_2_c2 = tf.layers.dropout(p_2_c2, rate=0.5)

    with tf.variable_scope('combine_parts'):
        combined = tf.concat([p_1_c2, p_2_c2], axis=-1)

        c = tf.layers.conv2d(combined, n3, kernel_size=1,
                activation=tf.nn.relu)
        c = tf.layers.dropout(c, rate=0.5)

    return tf.concat([c, X], axis=-1)


def create_env_model(obs_shape, num_actions, num_pixels, num_rewards,
        should_summary=True, reward_coeff=0.1):
    depth = obs_shape[0]
    width = obs_shape[1]
    height = obs_shape[2]

    states = tf.placeholder(tf.float32, [None, width, height, depth])

    onehot_actions = tf.placeholder(tf.float32, [None, width,
        height, num_actions]) 

    batch_size = tf.shape(states)[0]

    target_states = tf.placeholder(tf.uint8, [None])
    target_rewards = tf.placeholder(tf.uint8, [None])

    inputs = tf.concat([states, onehot_actions], axis=-1)

    with tf.variable_scope('pre_conv'):
        c = tf.layers.conv2d(inputs, 64, kernel_size=1, activation=tf.nn.relu)

    with tf.variable_scope('basic_block_1'):
        bb1 = basic_block(c, batch_size, 64, width, height, 16, 32, 64)

    with tf.variable_scope('basic_block_2'):
        bb2 = basic_block(bb1, batch_size, 128, width, height, 16, 32, 64)

    with tf.variable_scope('image_conver'):
        image = tf.layers.conv2d(bb2, 256, kernel_size=1, activation=tf.nn.relu)
        image = tf.layers.dropout(image, rate=0.5)
        image = tf.reshape(image, [batch_size * width * height, 256])
        image = tf.layers.dense(image, num_pixels)
        image = tf.layers.dropout(image, rate=0.5)

    with tf.variable_scope('reward'):
        reward = tf.layers.conv2d(bb2, 64, kernel_size=1,
                activation=tf.nn.relu)

        reward = tf.layers.conv2d(reward, 64, kernel_size=1,
                activation=tf.nn.relu)

        reward = tf.reshape(reward, [batch_size, width * height * 64])

        reward = tf.layers.dense(reward, num_rewards)

    target_states_one_hot = tf.one_hot(target_states, depth=num_pixels)
    image_loss = tf.losses.softmax_cross_entropy(target_states_one_hot, image)

    target_reward_one_hot = tf.one_hot(target_rewards, depth=num_rewards)
    reward_loss = tf.losses.softmax_cross_entropy(target_reward_one_hot, reward)

    loss = image_loss + (reward_coeff * reward_loss)

    opt = tf.train.AdamOptimizer().minimize(loss)

    # Tensorboard
    if should_summary:
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('reward_loss', reward_loss)
        tf.summary.scalar('image_loss', image_loss)

    return EnvModelData(image, reward, states, onehot_actions, loss,
            reward_loss, image_loss, target_states, target_rewards, opt)


def play_games(envs, frames):
    states = envs.reset()

    for frame_idx in range(frames):
        states = states.reshape(-1, nw, nh, nc)
        actions = [envs.action_space.sample() for _ in range(N_ENVS)]
        #actions, _, _ = actor_critic.act(states)
        #print(actions)
        next_states, rewards, dones, _ = envs.step(actions)

        yield frame_idx, states, actions, rewards, next_states, dones
        
        if(True in dones):
            #if int(frame_idx % 20) == 0: 
            #states = envs.reset()
            #rare_path = [2, 1, 3, 1, 3, 3, 0, 2, 1, 3, 1]
            #for action in rare_path:
            #    states = states.reshape(-1, nw, nh, nc)
            #    next_states, rewards, dones, _ = envs.step([action] * N_ENVS)
            #    yield frame_idx, states, action, rewards, next_states, dones
            #    states = next_states
            states = envs.reset()
        else:
            states = next_states


class EnvModelData(object):
    def __init__(self, imag_state, imag_reward, input_states, input_actions,
            loss, reward_loss, image_loss, target_states, target_rewards, opt):
        self.imag_state       = imag_state
        self.imag_reward      = imag_reward
        self.input_states     = input_states
        self.input_actions    = input_actions

        self.loss             = loss
        self.reward_loss      = reward_loss
        self.image_loss       = image_loss
        self.target_states    = target_states
        self.target_rewards   = target_rewards
        self.opt              = opt


if __name__ == '__main__':
    with tf.Session() as sess:
        #actor_critic = get_actor_critic(sess, N_ENVS, N_STEPS, ob_space, ac_space, CnnPolicy, should_summary=False)
        #actor_critic.load(A2C_WEIGHTS)

        with tf.variable_scope('env_model'):
            env_model = create_env_model(ob_space, num_actions, _NUM_PIXELS,
                    len(conveyer_rewards))

        sess.run(tf.global_variables_initializer())

        losses = []
        all_rewards = []

        depth = ob_space[0]
        width = ob_space[1]
        height = ob_space[2]

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
        saver = tf.train.Saver(var_list=save_vars)

        train_writer = tf.summary.FileWriter('./env_logs/train/', graph=sess.graph)
        val_writer = tf.summary.FileWriter('./env_logs/val/', graph=sess.graph)
        summary_op = tf.summary.merge_all()

        for frame_idx, states, actions, rewards, next_states, dones in tqdm(play_games(envs, NUM_UPDATES), total=NUM_UPDATES):
            target_state = pix_to_target(next_states)
            target_reward = rewards_to_target(rewards)

            # NOTE : which action at which point?
            onehot_actions = np.zeros((N_ENVS, num_actions, width, height))
            onehot_actions[range(N_ENVS), actions] = 1
            # Change so actions are the 'depth of the image' as tf expects
            onehot_actions = onehot_actions.transpose(0, 2, 3, 1)

            s, r, l, reward_loss, image_loss, summary, _ = sess.run([
                env_model.imag_state,
                env_model.imag_reward,
                env_model.loss,
                env_model.reward_loss,
                env_model.image_loss,
                summary_op,
                env_model.opt], feed_dict={
                    env_model.input_states: states,
                    env_model.input_actions: onehot_actions,
                    env_model.target_states: target_state,
                    env_model.target_rewards: target_reward
                })

            if frame_idx % LOG_INTERVAL == 0:
                print('%i => Loss : %.4f, Reward Loss : %.4f, Image Loss : %.4f' % (frame_idx, l, reward_loss, image_loss))
                
                validation_counter = 0
                '''
                for frame_idx, states, actions, rewards, next_states, dones in tqdm(play_games(actor_critic, envs, NUM_UPDATES), total=NUM_UPDATES):                    
                    val_target_state = pix_to_target(next_states)
                    val_target_reward = rewards_to_target(rewards)
                    val_onehot_actions = np.zeros((N_ENVS, num_actions, width, height))
                    val_onehot_actions[range(N_ENVS), val_actions] = 1
                    val_onehot_actions = val_onehot_actions.transpose(0, 2, 3, 1)

                    val_summary = sess.run(summary_op,
                        feed_dict={env_model.input_states: val_states, env_model.input_actions: val_onehot_actions, 
                        env_model.target_states: val_target_state, env_model.target_rewards: val_target_reward})

                    val_writer.add_summary(val_summary, validation_counter + 50 * (frame_idx) / LOG_INTERVAL)
                validation_counter += 1
                '''
            train_writer.add_summary(summary, frame_idx)

        saver.save(sess, 'weights/env_model.ckpt')
        print('Environment model saved!')

