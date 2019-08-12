import numpy as np

N_ENVS = 16

pixel_mapping = {
    'WALL_CHR'    : 0.0,
    ' '           : 1.0,
    'AGENT_CHR'   : 2.0,
    'OBJECT_CHR ' : 3.0,
    'END_CHR'     : 4.0,
    'BELT_CHR'    : 5.0,
    'GOAL_CHR'    : 6.0,
}

CONTROLS = ["UP", "DOWN", "LEFT", "RIGHT"]
_MAX_PIXEL_VAL = 5 
_NUM_PIXELS = _MAX_PIXEL_VAL + 1

# Source
# => ai-safety-gridworlds/environments/side_effects_sokoban

MOVEMENT_REWARD = 0
GOAL_REWARD = 50
REMOVAL_REWARD = GOAL_REWARD

conveyer_rewards = [MOVEMENT_REWARD, GOAL_REWARD, MOVEMENT_REWARD + GOAL_REWARD]

reward_to_categorical = {reward:i for i, reward in enumerate(conveyer_rewards)}

def pix_to_target(next_states):
    # Input shape : N_ENVS, 1, 6, 6
    return next_states.astype(int).flatten().tolist()

def target_to_pix(imagined_states):
    imagined_states = np.asarray(imagined_states, dtype=np.float32)
    #imagined_states = imagined_states.reshape(N_ENVS, 1, 6, 6)

    return imagined_states

def rewards_to_target(rewards):
    target = []
    if(isinstance(rewards, int) or isinstance(rewards, float)):
        rewards = [int(rewards)]
    for reward in rewards:
        target.append(reward_to_categorical[reward])
    return target
