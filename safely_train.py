# Inspired from OpenAI Baselines. This uses the same design of having an easily
# substitutable generic policy that can be trained. This allows to easily
# substitute in the I2A policy as opposed to the basic CNN one.

import copy
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from safe_grid_gym.envs.gridworlds_env import GridworldEnv

#from i2a import I2aPolicy
from utils import SubprocVecEnv
from discretize_env import CONTROLS
from a2c import CnnPolicy, get_actor_critic
from trajectory import generate_tree, search_node

ENV_NAME = "side_effects_sokoban"
N_ENVS = 16
N_STEPS = 9
END_REWARD = 49
S_ALPHA = 2
DEBUG = True

# Total number of iterations (taking into account number of environments and
# number of steps). You wish to train for.
TOTAL_TIMESTEPS = int(2e6)

GAMMA = 0.99

LOG_INTERVAL = 100
SAVE_INTERVAL = 100

# Where you want to save the weights
SAVE_PATH = 'weights'

def discount_with_dones(rewards, dones, GAMMA):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + GAMMA * r * (1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_env():
    def _thunk():
        env = GridworldEnv(ENV_NAME)
        return env

    return _thunk

def is_safe(trees, actions, base_state, dones):
    penal = []
    for i in range(len(actions)):
        if(dones[i] == True):
            penal.append(1)
            continue
        a = actions[i]
        next_node = trees[i].children[a]
        safe = search_node(next_node, base_state)
        if(next_node is not None and next_node.imagined_reward == END_REWARD):
            safe = True
        penal.append(int(safe))
        if(next_node is not None):
            trees[i] = next_node
    return penal, trees


def train(policy, save_name, load_count = 0, summarize=True, load_path=None, log_path = './logs'):
    envs = [make_env() for i in range(N_ENVS)]
    envs = SubprocVecEnv(envs)

    ob_space = envs.observation_space.shape
    nc, nw, nh = ob_space
    ac_space = envs.action_space

    obs = envs.reset()
    ob_np = np.copy(obs)
    ob_np = np.squeeze(ob_np, axis=1)
    ob_np = np.expand_dims(ob_np, axis=3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    actor_critic = get_actor_critic(sess, N_ENVS, N_STEPS, ob_space,
            ac_space, policy, summarize)
    if load_path is not None:
        actor_critic.load(load_path)
        print('Loaded a2c')

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_path, graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    batch_ob_shape = (N_ENVS * N_STEPS, nw, nh, nc)

    dones = [False for _ in range(N_ENVS)]
    nbatch = N_ENVS * N_STEPS

    episode_rewards = np.zeros((N_ENVS, ))
    final_rewards   = np.zeros((N_ENVS, ))

    # Safety part
    state = ob_np[0, :, :, :]

    base_state = copy.deepcopy(state).reshape(nc, nw, nh)
    base_state[np.where(base_state == 2.0)] = 1.0
    print(base_state)

    base_tree = generate_tree(sess, state)
    for update in tqdm(range(load_count + 1, TOTAL_TIMESTEPS + 1)):
        # mb stands for mini batch
        trees = [copy.deepcopy(base_tree)] * N_ENVS
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        for n in range(N_STEPS):
            ob_np = np.copy(obs)
            ob_np = np.squeeze(ob_np, axis=1)
            ob_np = np.expand_dims(ob_np, axis=3)

            '''
            for i in range(len(dones)):
                if(dones[i] == True):
                    trees[i] = copy.deepcopy(base_tree)
            '''

            if(update % LOG_INTERVAL == 0 and DEBUG == True):
                print_obs = ob_np[0, :, :, :].reshape(nc, nw, nh)
                print("-- State ---")
                print(print_obs)
                print("-- Imagined State --")
                print(trees[0].imagined_state.reshape(nc, nw, nh))

            actions, values, _ = actor_critic.act(ob_np)
            safe, trees = is_safe(trees, actions, base_state, dones)

            mb_obs.append(ob_np)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(dones)

            if(update % LOG_INTERVAL == 0 and DEBUG == True):
                print("Action : ", CONTROLS[actions[0]], " - Safe :", bool(safe[0])," - Done : ", dones[0])
                _ = input("")

            obs, rewards, dones, _ = envs.step(actions)

            rewards = [rewards[i] - S_ALPHA * (1 - safe[i]) for i in range(len(rewards))]
            episode_rewards += rewards
            masks = 1 - np.array(dones)
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            mb_rewards.append(rewards)

        mb_dones.append(dones)
        obs = envs.reset()

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(batch_ob_shape) #.swapaxes(1, 0).reshape(batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        last_values = actor_critic.critique(ob_np).tolist()

        #discount/bootstrap off value fn
        for n, (rewards, d, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            d = d.tolist()
            if d[-1] == 0:
                rewards = discount_with_dones(rewards+[value], d+[0], GAMMA)[:-1]
            else:
                rewards = discount_with_dones(rewards, d, GAMMA)
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
            print('%i => Policy Loss : %.4f, Value Loss : %.4f, Policy Entropy : %.4f, Final Reward : %.4f' % (update, policy_loss, value_loss, policy_entropy, final_rewards.mean()))

        if update % SAVE_INTERVAL == 0:
            print('Saving model')
            actor_critic.save(SAVE_PATH, save_name + '_' + str(update) + '.ckpt')

        actor_critic.save(SAVE_PATH, save_name + '_done.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', help='Algorithm to train a2c (or something else in the future)')
    args = parser.parse_args()

    if args.algo == 'a2c':
        policy = CnnPolicy
    else:
        raise ValueError('Must specify the algo name as either a2c or (something else in the future)')

    train(policy, args.algo, summarize=True, log_path=args.algo + '_logs')

