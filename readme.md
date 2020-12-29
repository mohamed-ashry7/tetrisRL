# Tetris Engine for Reinforcement Learning

This Library is implementing the Tetris Environment and adapt it to be used in the RL agents. 


## Layout

* tetris_engine.py - this file contains the main class of Tetris engine. 
* controllers.py - this file contains many controllers that help to evaluate the states and the reward functions. For more info check [this](https://www.researchgate.net/publication/345851349_Applying_Deep_Q-Networks_DQN_to_the_game_of_Tetris_using_high-level_state_spaces_and_different_reward_func-_tions)


## Usage


```python
from tetris_engine import TetrisEngine
 
width,height = 10,20
env = TetrisEngine(width,height)

```

This engine is mainly designed for RL agents. It takes the OpenAI interface
```python

while True:
    # Get an action from a theoretical AI agent
    action = agent(state)

    # Sim step takes action and returns results
    next_state, reward, done = env.step(action)

    if done:
        break

```

This engine is integrated into a [dqn_agent](https://github.com/mohamed-ashry7/tetrisRL)