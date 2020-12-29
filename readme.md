# Tetris RL Agent

This repo is implementing an RL agent integrated with Tetris engine to try to create a DQN agent able to learn how to play Tetris game. 

It is implementing the agents and the controllers mentioned in that [thesis](https://www.researchgate.net/publication/345851349_Applying_Deep_Q-Networks_DQN_to_the_game_of_Tetris_using_high-level_state_spaces_and_different_reward_func-_tions)



## Layout
* tetris_agent.py - It contains the RL agent and manipulate the connection between the DQN and the environment. 
* run_agent.py - This is the run file that contains also the most of the hyperparameters.
* replay_buffer.py - Please refer to [thesis](https://www.researchgate.net/publication/345851349_Applying_Deep_Q-Networks_DQN_to_the_game_of_Tetris_using_high-level_state_spaces_and_different_reward_func-_tions)
* tetris_engine.py - This file contains the main class of Tetris engine. 
* controllers.py - This file contains many controllers that help to evaluate the states and the reward functions. For more info check 
* models
    - dqn_model.py - It is the NN that receives the data. 
    - noisy_layer.py - It is a custom NN layer that is used in DQN NN. 
* utils
    - drawing_utils.py - It contains functions to draw and plot the data. 
    - tetris_engine_utils.py - It contains the necessary functions for the Tetris Engine.
* dqn_models_stats - It is the directory that store the saved models from the NN. 

## Usage

Run the run_agent.py file with optional arguments
```bash
python run_agent.py

```
The arguments are 
```
--cuda # To decide on what the model shall be trained. CPU or GPU.
--width # The width of the board the default is 10.
--height # The width of the board the default is 20.
--mode # Whether it is play or train mode. play mode here means to test the NN. 
--model # The name of the model which be loaded to train from or to test it.
--model-dir # The path of the directory where the models will be saved. default=dqn_models_stats.
--play-times # Number of games to be played to test the NN. default= 100.
```
