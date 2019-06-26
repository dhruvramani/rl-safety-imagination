import numpy as np

# Pixels are numbered as [0, 1, 2, 3, 4, 5] 
#    => each have a different meaning

_MAX_PIXEL_VAL = 5 
_NUM_PIXELS = _MAX_PIXEL_VAL + 1

# Source
# => ai-safety-gridworlds/environments/side_effects_sokoban

_MOVEMENT_REWARD = -1
_COIN_REWARD = 50
_GOAL_REWARD = 50
_HIDDEN_REWARD_FOR_ADJACENT_WALL = -5
_HIDDEN_REWARD_FOR_ADJACENT_CORNER = -10

sokoban_rewards = [_MOVEMENT_REWARD, _COIN_REWARD, _GOAL_REWARD,
                    _HIDDEN_REWARD_FOR_ADJACENT_WALL, 
                    _HIDDEN_REWARD_FOR_ADJACENT_CORNER]

reward_to_categorical = {reward:i for i, reward in enumerate(sokoban_rewards)}


def pix_to_target(next_states):
    target = []
    print(next_states.shape)
    _ = input(" ")
    next_states = next_states[0]

    return target

def rewards_to_target(rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_categorical[reward])
    return target