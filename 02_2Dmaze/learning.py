"""
This is a simple maze example for reinforcement learning
There is an agent on a 2D maze that needs to get to the finish square
Blue rectangle: agent
Black rectangles: penalty states (reward = -1]
Yellow rectangle: Finish [reward = 1]
All other rectangles: nothing [reward = 0]
"""

import numpy as np
import pandas as pd

# Define hyperparameters of the learning
learning_rate = 0.01
reward_decay = 0.9
e_greedy = 0.9

# Define our Q_learning setting as a class with a table
class QLearning:
    # Constructor
    def __init__(self, actions, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy):
        self.actions = actions # List of possible actions
        self.lr = learning_rate # Learning rate
        self.gamma = reward_decay # Gamma value (reward decay)
        self.epsilon = e_greedy # Epsilon, how greedy we are
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # The Q-table, columns are available actions

    # Choose action function
    def choose_action(self, observation):
        self.check_state_exist(observation) # Check if the state exists
        if np.random.uniform() < self.epsilon: # Select greedy action
            state_action = self.q_table.loc[observation, :] # choose best action
            action = np.random.choice(state_action[state_action == np.max(state_action)].index) # Random choose if more than 1
        else: # Select random action
            action = np.random.choice(self.actions)
        return action

    # Learn function
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_) # Check if the state exists
        q_predict = self.q_table.loc[s, a] # Get the current Q-value of state and action
        if s_ != 'terminal': # If the new state is not terminal
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else: # New state is terminal
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # Update table value

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )