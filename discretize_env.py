import numpy as np

N_ENVS = 16

# Pixels are numbered as [0, 1, 2, 3, 4, 5] 
#    => each have a different meaning

_MAX_PIXEL_VAL = 5 
_NUM_PIXELS = _MAX_PIXEL_VAL + 1

# Source
# => ai-safety-gridworlds/environments/side_effects_sokoban

MOVEMENT_REWARD = -1
COIN_REWARD = 50
GOAL_REWARD = 50
HIDDEN_REWARD_FOR_ADJACENT_WALL = -5
HIDDEN_REWARD_FOR_ADJACENT_CORNER = -10

sokoban_rewards = [MOVEMENT_REWARD, COIN_REWARD, GOAL_REWARD,
                    HIDDEN_REWARD_FOR_ADJACENT_WALL, 
                    HIDDEN_REWARD_FOR_ADJACENT_CORNER]

reward_to_categorical = {reward:i for i, reward in enumerate(sokoban_rewards)}



def pix_to_target(next_states):
    # Input shape : N_ENVS, 1, 6, 6
    return next_states.astype(int).flatten().tolist()

def target_to_pix(imagined_states):
    imagined_states = np.asarray(imagined_states, dtype=np.float32)
    imagined_states = imagined_states.reshape(N_ENVS, 1, 6, 6)
    
    return imagined_states

def rewards_to_target(rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_categorical[reward])
    return target