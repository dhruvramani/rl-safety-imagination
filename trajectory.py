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
from discretize_env import pix_to_target, target_to_pix, rewards_to_target, _NUM_PIXELS, CONTROLS, conveyer_rewards

# TODO - REMOVE ACTOR CRITIC WHERE NOT REQUIRED
N_ENVS = 1
N_STEPS = 5
END_REWARD = 49
MAX_TREE_STEPS = 9
NUM_ROLLOUTS = 10 # Hyperparameter of how far ahead in the future the agent "imagines"
DEBUG = False
 
A2C_MODEL_PATH = 'weights/a2c_3600.ckpt'
ENV_MODEL_PATH = 'weights/env_model.ckpt'

np.set_printoptions(threshold=sys.maxsize)

envs = [make_env() for i in range(N_ENVS)]
envs = SubprocVecEnv(envs)
ob_space = envs.observation_space.shape
ac_space = envs.action_space
num_actions = ac_space.n
num_rewards = len(conveyer_rewards)
nc, nw, nh = ob_space

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
                    len(conveyer_rewards), should_summary=False)

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
        loader = tf.train.Saver(var_list=save_vars)
        loader.restore(sess, ENV_MODEL_PATH)

        print('Env model restored!')

    return g_env_model

'''
def generate_trajectory(sess, state):

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
        imagined_rewards_list.append(conveyer_rewards[imagined_reward[0]])
        if(conveyer_rewards[imagined_reward[0]] == END_REWARD):
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

def search_node(root, unsafe_state, debug=False):
    if(root is not None):
        imagined_state = copy.deepcopy(root.imagined_state)
        imagined_state = imagined_state.reshape(nc, nw, nh)
        index = np.where(imagined_state == 2.0)
        try :
            if(imagined_state[index[0][0], index[1][0], index[2][0] + 1] == 5.0 or 
                imagined_state[index[0][0], index[1][0], index[2][0] - 1] == 5.0):
                imagined_state[index] = 5.0
            else : 
                imagined_state[index] = 1.0
        except:
            return True
        # NOTE : Check if it ends with a safe state
        if(np.array_equal(imagined_state, unsafe_state) and root.children == [None, None, None, None]):
            return False
        for child in root.children:
            found = search_node(child, unsafe_state, debug=debug)
            if(found == False):
                return found
    return True

def generate_tree(sess, state, reward=-1, count=0):
    # TODO : Recursion count 1, allow END REWARD
    nc, nw, nh = ob_space
    num_actions = ac_space.n
    num_rewards = len(conveyer_rewards)

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
        imagined_state, imagined_reward = imagined_states[0][0, 0, :, :], conveyer_rewards[np.argmax(imagined_rewards[0], axis=1)[0]]
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

def get_node(root, state):
    state = state.reshape(nc, nw, nh)
    current_node = copy.deepcopy(root)
    queue = []
    queue.append(current_node)
    while len(queue) != 0:
        current_node = queue.pop(0)
        curr_state = copy.deepcopy(current_node.imagined_state)
        curr_state = curr_state.reshape(nc, nw, nh)

        if(np.array_equal(curr_state, state)):
            return current_node

        for child in current_node.children:
            if(child is not None):
                queue.append(child)

    return None

def act_safely(sess):
    env = GridworldEnv("conveyor_belt")
    num_actions = ac_space.n
    num_rewards = len(conveyer_rewards)

    actor_critic = get_cache_loaded_a2c(sess, N_ENVS, N_STEPS, ob_space, ac_space)
    state = env.reset()
    base_state = copy.deepcopy(state)
    base_state = base_state.reshape(nc, nw, nh)
    base_state[np.where(base_state == 2.0)] = 1.0
    print(base_state)

    root = generate_tree(sess, state)
    tree = copy.deepcopy(root)
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
        if(DEBUG):
            print("-- Current State --")
            print(state)
        if(is_end == False and search_node(next_node, base_state) == True):
            old_action = CONTROLS[actions[0]]
            actions = safe_action(actor_critic, tree, base_state, actions[0])
            if(DEBUG):
                print("Unsafe - Old Action : ", old_action, end="")
                print("- New Action : ", CONTROLS[actions[0]])
        state, reward, done, _ = env.step(actions[0])
        if(not DEBUG):
            env.render()
        tree = get_node(root, state) #tree.children[actions[0]]
        steps += 1

def plot_predictions(sess):

    BAD_STATES = [np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 4.0, 1.0, 2.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0, 5.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
               np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 4.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 2.0, 5.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),]

    env = GridworldEnv("conveyor_belt")
    num_actions = ac_space.n
    nc, nw, nh = ob_space
    num_rewards = len(conveyer_rewards)

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


if __name__ == '__main__':
    env = GridworldEnv("conveyor_belt")
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
    
       
    # TREEEEEE lol
    
    state = envs.reset()
    base_state = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 5.0, 5.0, 5.0, 5.0, 4.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    base_state = base_state.reshape(nc, nw, nh)
    print(base_state)

    print("=> Generating Tree")
    root = generate_tree(sess, ob_np)
    node = copy.deepcopy(root)
    path = [1, 1, 2, 1, 1, 3, 0, 0, 0, 0, 1, 0]

    count = 0
    done = False
    debug = False
    while(done != True):
        next_node = node.children[path[count]]
        print("-- Current State --")
        print(state)
        search = search_node(next_node, base_state, debug=debug)
        if(search == False):
            print("Unsafe Action : ", CONTROLS[path[count]])
        state, _, done, _ = env.step(path[count])
        #env.render("human", imagined_state, node.imagined_reward)
        #print(node.imagined_state.reshape(nc, nw, nh))
        node = get_node(root, node.children[path[count]].imagined_state)
        count += 1

    '''
    # NOT-SO-TREEEEE lolol
    imagined_states, imagined_rewards = generate_trajectory(sess, ob_np)

    for i in range(len(imagined_states)):
        _, _, _, _ = env.step(ac_space.sample())
        #print(imagined_states[i], imagined_rewards[i])
        env.render("human", imagined_states[i], imagined_rewards[i])
        time.sleep(0.4)
    
    '''
