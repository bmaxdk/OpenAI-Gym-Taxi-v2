# Taxi Problem

### Getting Started

Read the description of the environment in subsection 3.1 of [this paper](https://arxiv.org/pdf/cs/9905014.pdf).  You can verify that the description in the paper matches the OpenAI Gym environment by peeking at the code [here](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py).


### Instructions

The repository contains three files:
- `agent.py`: Develop your reinforcement learning agent here. 
- `monitor.py`: The `interact` function tests how well your agent learns from interaction with the environment.
- `main.py`: Run this file in the terminal to check the performance of your agent.

Begin by running the following command in the terminal:
```bash
$ python main.py
```

When you run `main.py`, the agent that you specify in `agent.py` interacts with the environment for 20,000 episodes.  The details of the interaction are specified in `monitor.py`, which returns two variables: `avg_rewards` and `best_avg_reward`.
- `avg_rewards` is a deque where `avg_rewards[i]` is the average (undiscounted) return collected by the agent from episodes `i+1` to episode `i+100`, inclusive.  So, for instance, `avg_rewards[0]` is the average return collected by the agent over the first 100 episodes.
- `best_avg_reward` is the largest entry in `avg_rewards`.  This is the final score that you should use when determining how well your agent performed in the task.

Your assignment is to modify the `agents.py` file to improve the agent's performance.
- Use the `__init__()` method to define any needed instance variables.  Currently, we define the number of actions available to the agent (`nA`) and initialize the action values (`Q`) to an empty dictionary of arrays.  Feel free to add more instance variables; for example, you may find it useful to define the value of epsilon if the agent uses an epsilon-greedy policy for selecting actions.
- The `select_action()` method accepts the environment state as input and returns the agent's choice of action.  The default code that we have provided randomly selects an action.
- The `step()` method accepts a (`state`, `action`, `reward`, `next_state`) tuple as input, along with the `done` variable, which is `True` if the episode has ended.  The default code (which you should certainly change!) increments the action value of the previous state-action pair by 1.  You should change this method to use the sampled tuple of experience to update the agent's knowledge of the problem.

Once you have modified the function, you need only run `python main.py` to test your new agent.

OpenAI Gym [defines "solving"](https://gym.openai.com/envs/Taxi-v1/) this task as getting average return of 9.7 over 100 consecutive trials.









# OpenAI-Gym-Taxi-v3
[here](https://github.com/bmaxdk/OpenAI-Gym-Taxi-v3.git)
## Attempts to solve OpenAI Gym's Taxi-v3 RL environment

The skeleton of this code is from [Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/lab-taxi).  Their version uses Taxi-v2, but this version uses v3, since v2 is deprecated.

The environment is from [here](https://gym.openai.com/envs/Taxi-v3/).

To do the simple demo, on Linux or Mac with Docker installed, make `taxi.sh` executable and run it:
```bash
$ git clone https://github.com/bmaxdk/OpenAI-Gym-Taxi-v2.git
$ cd OpenAI-Gym-Taxi-v2
$ python main.py
```

This version uses a variation on standard Q-learning.  The policy is epsilon-greedy, but when the non-greedy action is chosen, instead of being sampled from a uniform distribution, it is sampled from a distribution that reflects two things:
   - a preference for actions with higher Q values (i.e. "greedy but flexible")
   - a preference for novel actions (those that have recently been less often chosen in the current state)

The latter are tracked via a "path memory" table (same shape as the Q table), which counts how often each action is taken in each state.  At the end of each episode, path memories from the previous episode decay geometrically.  

The sampling distribution for stochastic actions is the softmax of a linear combination of the Q values (with a positive coefficient) and the path memory values (with a negative coefficient).







# Result
python main.py 
Episode 20000/20000 || Best average reward 9.0437
