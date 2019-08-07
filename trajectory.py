import os
import sys
import gym
import time
import copy
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve 

from utils import SubprocVecEnv
from imagine import ImaginationCore
from a2c import get_actor_critic, CnnPolicy
from env_model import make_env, create_env_model
from safe_grid_gym.envs.gridworlds_env import GridworldEnv
from discretize_env import pix_to_target, target_to_pix, rewards_to_target, _NUM_PIXELS, CONTROLS, sokoban_rewards

# TODO - REMOVE ACTOR CRITIC WHERE NOT REQUIRED
N_ENVS = 1
N_STEPS = 5
END_REWARD = 49
MAX_TREE_STEPS = 10
NUM_ROLLOUTS = 10 # Hyperparameter of how far ahead in the future the agent "imagines"
 
A2C_MODEL_PATH = 'weights/a2c_3600.ckpt'
ENV_MODEL_PATH = 'weights/env_model.ckpt'

np.set_printoptions(threshold=sys.maxsize)

envs = [make_env() for i in range(N_ENVS)]
envs = SubprocVecEnv(envs)
ob_space = envs.observation_space.shape
ac_space = envs.action_space
nc, nw, nh = ob_space

#FINAL_STATE = []
BAD_STATES = [#np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.],
#                           [0.0, 1.0, 1.0, 0.0, 0.0, 0.],
#                           [0.0, 1.0, 2.0, 1.0, 1.0, 0.],
#                           [0.0, 0.0, 4.0, 1.0, 1.0, 0.],
#                           [0.0, 0.0, 0.0, 1.0, 5.0, 0.],
#                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.]]),
#                np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.],
#                            [0.0, 1.0, 1.0, 0.0, 0.0, 0.],
#                            [0.0, 1.0, 1.0, 2.0, 1.0, 0.],
#                            [0.0, 0.0, 4.0, 1.0, 1.0, 0.],
#                            [0.0, 0.0, 0.0, 1.0, 5.0, 0.],
#                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.]]),
#                np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.],
#                            [0.0, 1.0, 1.0, 0.0, 0.0, 0.],
#                            [0.0, 1.0, 1.0, 1.0, 2.0, 0.],
#                            [0.0, 0.0, 4.0, 1.0, 1.0, 0.],
#                            [0.0, 0.0, 0.0, 1.0, 5.0, 0.],
#                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.]]),
               np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.],
                           [0.0, 1.0, 1.0, 0.0, 0.0, 0.],
                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.],
                           [0.0, 0.0, 4.0, 1.0, 2.0, 0.],
                           [0.0, 0.0, 0.0, 1.0, 5.0, 0.],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.]]),
               # np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.],
               #             [0.0, 1.0, 1.0, 0.0, 0.0, 0.],
               #             [0.0, 1.0, 1.0, 1.0, 1.0, 0.],
               #             [0.0, 0.0, 4.0, 2.0, 1.0, 0.],
               #             [0.0, 0.0, 0.0, 1.0, 5.0, 0.],
               #             [0.0, 0.0, 0.0, 0.0, 0.0, 0.]]),
               np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.],
                           [0.0, 1.0, 1.0, 0.0, 0.0, 0.],
                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.],
                           [0.0, 0.0, 4.0, 1.0, 1.0, 0.],
                           [0.0, 0.0, 0.0, 2.0, 5.0, 0.],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.]]),]



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

'''
def generate_trajectory(sess, state):
    num_actions = ac_space.n
    num_rewards = len(sokoban_rewards)

    actor_critic = get_cache_loaded_a2c(sess, N_ENVS, N_STEPS, ob_space, ac_space)
    env_model = get_cache_loaded_env_model(sess, ob_space, num_actions)

    imagination = ImaginationCore(NUM_ROLLOUTS, num_actions, num_rewards,
                ob_space, actor_critic, env_model)

    imagined_states, imagined_rewards = imagination.imagine(state, sess)
    imagined_states_list, imagined_rewards_list = [],  []
    for i in range(len(imagined_states)):
        imagined_state, imagined_reward = imagined_states[i], imagined_rewards[i]
        imagined_reward = np.argmax(imagined_reward, axis=1)

        #for i in range(imagined_states.shape[0]):
        imagined_states_list.append(imagined_state[0, 0, :, :])
        imagined_rewards_list.append(sokoban_rewards[imagined_reward[0]])
        if(sokoban_rewards[imagined_reward[0]] == END_REWARD):
            break

    return imagined_states_list, imagined_rewards_list
'''

class ImaginedNode(object):
    def __init__(self, imagined_state, imagined_reward):
        self.imagined_state  = imagined_state
        self.imagined_reward = imagined_reward
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

def search_node(root, base_state):
    if(root is not None):
        #print(root.imagined_state.reshape(nc, nw, nh))
        imagined_state = copy.deepcopy(root.imagined_state)
        imagined_state = imagined_state.reshape(nc, nw, nh)
        imagined_state[np.where(imagined_state == 2.0)] = 1.0
        if(np.array_equal(imagined_state, base_state) and root.imagined_reward != END_REWARD):
            return True
        for child in root.children:
            found = search_node(child, base_state)
            if(found == True):
                return found
    return False

def generate_tree(sess, state, reward=-1, count=0):
    # TODO : Recursion count 1, allow END REWARD
    nc, nw, nh = ob_space
    num_actions = ac_space.n
    num_rewards = len(sokoban_rewards)

    env_model = get_cache_loaded_env_model(sess, ob_space, num_actions)

    imagination = ImaginationCore(num_actions, num_rewards,
                ob_space, env_model)

    state = state.reshape(-1, nw, nh, nc)
    node = ImaginedNode(state, reward)
    if(reward == END_REWARD or count > MAX_TREE_STEPS):
        node.children.extend([None for i in range(num_actions)])
        return node

    for action in range(num_actions):
        imagined_states, imagined_rewards = imagination.imagine(state, sess, action)
        imagined_state, imagined_reward = imagined_states[0][0, 0, :, :], sokoban_rewards[np.argmax(imagined_rewards[0], axis=1)[0]]
        if(np.array_equal(state.reshape(nw, nh), imagined_state)):
            node.add_child(None)
            continue
        imagined_state = imagined_state.reshape(-1, nw, nh, nc)
        node.add_child(generate_tree(sess, imagined_state, imagined_reward, count + 1))

    return node

def safe_action(agent, tree, base_state, unsafe_action):
    possible_actions = [i for i in range(ac_space.n) if i != unsafe_action]
    imagined_states =  {a : tree.children[a].imagined_state for a in possible_actions if search_node(tree.children[a], base_state) == True}
    values = {a : agent.critique(imagined_states[a].reshape(1, nw, nh, nc)) for a in imagined_states.keys()}
    for a in possible_actions:
        if(tree.children[a] is not None and tree.children[a].imagined_reward == END_REWARD):
            values[a] = END_REWARD
    max_a = max(values.keys(), key=lambda a:values[a])
    return [max_a]

def next_node(root, state):
    current_node = copy.deepcopy(root)
    queue, found = [], False
    queue.append(current_node)
    while found == False and len(queue) != 0:
        curr_state = copy.deepcopy()

# NOTE : Uncomment after getting proper A2C weights
def act_safely(sess):
    env = GridworldEnv("side_effects_sokoban")
    num_actions = ac_space.n
    num_rewards = len(sokoban_rewards)

    actor_critic = get_cache_loaded_a2c(sess, N_ENVS, N_STEPS, ob_space, ac_space)
    state = env.reset()
    base_state = copy.deepcopy(state)
    base_state = base_state.reshape(nc, nw, nh)
    base_state[np.where(base_state == 2.0)] = 1.0
    print(base_state)

    tree = generate_tree(sess, state)
    print("Tree Created")
    done, steps = False, 0
    
    while(done != True):
        actions, _, _ = actor_critic.act(np.expand_dims(state, axis=3))
        is_end = False
        try :
            next_node = tree.children[actions[0]]
            is_end = next_node.imagined_reward == END_REWARD
        except AttributeError:
            next_node = None 
        print("-- Current State --")
        print(state)
        if(is_end == False and search_node(next_node, base_state) == False):
            print("Unsafe - Old Action : ", CONTROLS[actions[0]], end="")
            actions = safe_action(actor_critic, tree, base_state, actions[0])
            print("- New Action : ", CONTROLS[actions[0]])
        state, reward, done, _ = env.step(actions[0])
        #env.render()
        tree = tree.children[actions[0]]
        steps += 1
'''
def plot_predictions(sess):
    env = GridworldEnv("side_effects_sokoban")
    num_actions = ac_space.n
    nc, nw, nh = ob_space
    num_rewards = len(sokoban_rewards)

    actor_critic = get_cache_loaded_a2c(sess, N_ENVS, N_STEPS, ob_space, ac_space)

    state = env.reset()
    done, steps = False, 0
    labels, predictions = [], []

    while done != True and steps < NUM_ROLLOUTS:
        imagine_rollouts, _ = generate_trajectory(sess, state)
        is_bad = False
        for bad_state in BAD_STATES:
            for imagined_state in imagine_rollouts:
                if (np.array_equal(bad_state, imagined_state)):
                    is_bad = True
                    break

        if(is_bad == True):
            predictions += [[1.0, 0.0]]
        else :
            predictions += [[0.0, 1.0]]

        is_bad = False
        for bad_state in BAD_STATES:
            if (np.array_equal(state.reshape(nw, nh), bad_state)):
                is_bad = True
                break

        if(is_bad == True):
            labels = [[1.0, 0.0]] * (steps + 1)
        else :
            labels += [[0.0, 1.0]]

        actions, _, _ = actor_critic.act(np.expand_dims(state, axis=3))
        state, reward, done, _ = env.step(actions[0])
        steps += 1

    labels += [[0.0, 1.0]]
    predictions += [[0.0, 1.0]]
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    print("ROC AUC Score : ", roc_auc_score(labels, predictions))
    #print("Precision Recall Curve : ", precision_recall_curve(labels, predictions))
'''

if __name__ == '__main__':
    env = GridworldEnv("side_effects_sokoban")
    env.reset()

    nc, nw, nh = ob_space

    obs = envs.reset()
    ob_np = np.copy(obs)
    ob_np = np.squeeze(ob_np, axis=1)
    ob_np = np.expand_dims(ob_np, axis=3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #act_safely(sess)
    #plot_predictions(sess)
    
    '''    
    # TREEEEEE lol
    
    state = envs.reset()
    base_state = np.copy(state)
    base_state = base_state.reshape(nc, nw, nh)
    base_state[np.where(base_state == 2.0)] = 1.0
    print(base_state)

    print("=> Generating Tree")
    node = generate_tree(sess, ob_np)

    #path = [2, 1, 3, 1, 3, 0, 1, 3]
    path = [2, 1, 3, 1, 3, 3, 0, 2, 1, 3, 1]
    #path = [1, 3, 3, 1, 1]
    count = 0
    done = False
    while(done != True):
        next_node = node.children[path[count]]
        print("-- Current State --")
        print(state)
        if(search_node(next_node, base_state) == False):
            print("Unsafe Action : ", CONTROLS[path[count]])
        _ = input(" ")
        state, _, done, _ = env.step(path[count])
        #env.render("human", imagined_state, node.imagined_reward)
        node = node.children[path[count]]
        count += 1
    '''
    '''
    # NOT-SO-TREEEEE lolol
    imagined_states, imagined_rewards = generate_trajectory(sess, ob_np)

    for i in range(len(imagined_states)):
        _, _, _, _ = env.step(ac_space.sample())
        #print(imagined_states[i], imagined_rewards[i])
        env.render("human", imagined_states[i], imagined_rewards[i])
        time.sleep(0.4)
    
    '''
