# Inspired from OpenAI Baselines. This uses the same design of having an easily
# substitutable generic policy that can be trained. This allows to easily
# substitute in the I2A policy as opposed to the basic CNN one.

import gym
import argparse
import numpy as np
import safe_grid_gym
from tqdm import tqdm
import tensorflow as tf

#from common.minipacman import MiniPacman
#from common.multiprocessing_env import SubprocVecEnv
#from i2a import I2aPolicy

from a2c import CnnPolicy, get_actor_critic



N_ENVS  = 1
N_STEPS = 10

# Total number of iterations (taking into account number of environments and
# number of steps). You wish to train for.
TOTAL_TIMESTEPS=int(1e6)

GAMMA=0.99

LOG_INTERVAL=100
SAVE_INTERVAL = 1e5

# Where you want to save the weights
SAVE_PATH = 'weights'

# This can be anything from "regular" "avoid" "hunt" "ambush" "rush" each
# resulting in a different reward function giving the agent different behavior.
REWARD_MODE = 'regular'

def discount_with_done(rewards, done, GAMMA):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], done[::-1]):
        r = reward + GAMMA * r * (1. - done)
        discounted.append(r)
    return discounted[::-1]

def make_env():
    def _thunk():
        env = gym.make("SideEffectsSokoban-v0")
        return env

    return _thunk

def train(policy, save_name, load_count = 0, summarize=True, load_path=None, log_path = './logs'):
    env = make_env()()

    ob_space = env.observation_space.shape
    nc, nw, nh = ob_space
    ac_space = env.action_space

    obs = env.reset()
    ob_np = np.copy(obs)
    ob_np = np.expand_dims(ob_np, axis=3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    actor_critic = get_actor_critic(sess, N_STEPS, ob_space,
            ac_space, policy, summarize)
    if load_path is not None:
        actor_critic.load(load_path)
        print('Loaded a2c')

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_path, graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    batch_ob_shape = (N_STEPS, nw, nh, nc)

    done = False
    nbatch = N_STEPS

    episode_rewards = 0
    final_rewards   = 0

    for update in tqdm(range(load_count + 1, TOTAL_TIMESTEPS + 1)):
        # mb stands for mini batch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_done = [],[],[],[],[]
        for n in range(N_STEPS):
            ob_np = np.copy(obs)
            ob_np = np.expand_dims(ob_np, axis=3)
            actions, values, _ = actor_critic.act(ob_np)

            mb_obs.append(ob_np)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_done.append(done)

            obs, rewards, done, _ = env.step(actions)

            # TODO - SEE Properly
            episode_rewards += rewards
            masks = 1 - int(done)
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            mb_rewards.append(rewards)

        mb_done.append(done)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(batch_ob_shape) #.swapaxes(1, 0).
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)#.swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)#.swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32)#.swapaxes(1, 0)
        mb_done = np.asarray(mb_done, dtype=np.bool)#.swapaxes(1, 0)
        mb_masks = mb_done[:-1]
        mb_done = mb_done[1:]

        last_values = actor_critic.critique(ob_np).tolist()

        #discount/bootstrap off value fn
        for n, (rewards, d, value) in enumerate(zip(mb_rewards, mb_done, last_values)):
            rewards = mb_rewards.tolist()
            if d == 0:
                rewards = discount_with_done(rewards+[value], mb_done, GAMMA)[:-1]
            else:
                rewards = discount_with_done(rewards, mb_done, GAMMA)
            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        if summarize:
            loss, policy_loss, value_loss, policy_entropy, _, summary = actor_critic.train(mb_obs,
                    mb_rewards, mb_masks, mb_actions, mb_values, update,
                    summary_op)
            writer.add_summary(summary, update)
        else:
            loss, policy_loss, value_loss, policy_entropy, _ = actor_critic.train(mb_obs,
                    mb_rewards, mb_masks, mb_actions, mb_values, update)

        if update % LOG_INTERVAL == 0 or update == 1:
            print('%i): %.4f, %.4f, %.4f' % (update, policy_loss, value_loss, policy_entropy))
            print(final_rewards.mean())

        if update % SAVE_INTERVAL == 0:
            print('Saving model')
            actor_critic.save(SAVE_PATH, save_name + '_' + str(update) + '.ckpt')

        actor_critic.save(SAVE_PATH, save_name + '_done.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', help='Algorithm to train either i2a or a2c')
    args = parser.parse_args()

    if args.algo == 'a2c':
        policy = CnnPolicy
    elif args.algo == 'i2a':
        policy = I2aPolicy
    else:
        raise ValueError('Must specify the algo name as either a2c or i2a')

    env = make_env()()
    #ob_space = env.observation_space.shape
    #ac_space = env.action_space
    #print(ob_space, ac_space)
    train(policy, args.algo, summarize=True, log_path=args.algo + '_logs')
