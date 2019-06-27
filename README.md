# Avoiding Negative Side-Effects and Promoting Safe Exploration Using Imaginative Planning

All of the experimentation is done on AI-Safety-Gridworlds environment. Currently, the code doesn't explicity consider any safety, acts like a base we'll improve upon.

## Model 
This model (currently) consists of 2 networks (sub-models) - the actor critic (A2C) and an environment model (env_model).
The A2C is a standard one, self-explanatory. The env_model takes the current state and the action predicted by the policy of A2C and predicts the next state and the reward obtained. To generate the imaginative trajectory - we run this in a loop with the next input as the current output (`NOTE: needs discussion, can/should use some form of RNN`).

## Installation
Other than your standard ML/RL ammunition, you'll need to install a Gym wrapper for the original [environments](https://github.com/deepmind/ai-safety-gridworlds) by DeepMind. Thanks to [@david-lindner](https://github.com/david-lindner), 
```shell
git clone https://github.com/david-lindner/safe-grid-gym
cd safe-grid-gym
python3 setup.py install
```

## Loading and Testing
Download the logs and the weights from [here](https://drive.google.com/drive/folders/1-IPWRcXzoVy1g_rNBqiEwnwpz6uELqzN?usp=sharing). To test the A2C and Environment Model, use `eval_actor_vis.ipynb` and `eval_env_model.ipynb` respectively. The imaginative trajectory can be obtained from `trajectory.py`. Running it directly will print the imagined sates (`NOTE : buggy, fix this`). 

## Training
To train from the starting, first train the A2C (`NOTE : If it starts with -100 rewards - retrain. Needs early stopping`).  
```
python3 onpolicy_train.py --algo a2c
```
Using the saved A2C, train env_model 
```
python3 env_model.py
```
