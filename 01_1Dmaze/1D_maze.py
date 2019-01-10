"""""
This is the simplest possible Q learning example where we have an agent
that learns the best way to get to the finish (right extreme) of a 1D 
discrete sequence of locations.

The environment is represented by 'x', the agent as 'O' and the finish
point as 'F', and depending on the selected size of the game it will look
as this: xxxxxOxxxxF, where the 'O' can move left or right trying to find F.
"""""

import numpy as np
import pandas as pd
import time

NUM_STATES = 10   # This is the number discrete states that our universe has
ACTIONS = ['left', 'right']     # The available actions of the agent
EPSILON = 0.8   # This is how greedy we are when choosing the optimal step against exploration
ALPHA = 0.1     # This is our learning rate for Q table updates
GAMMA = 0.9    # This is the discount factor of feature reward
MAX_EPISODES = 13   # This is the number of games we are going to play
TIME_STEP = 0.3    # This is the printing rate so we can see what is happening

# We build a table with as many rows as states and as many columns as available actions
def build_q_table(n_states, actions):
    initial_values = np.zeros((n_states, len(actions))) # Initial values of the table
    table = pd.DataFrame(initial_values, columns=actions) # Build table as dataframe
    return table

# This function chooses an action to take based on Q table and greedy policy (EPSILON)
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :] # Get all avialable actions for our current state
    # Choose random action with probability 1-EPSILON or if we know nothing about the actions given the current state
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    # Choose the action that maximizes future reward based on Q table
    else:
        action_name = state_actions.idxmax()
    return action_name

# Read the environment to get feedback
def get_env_feedback(state, action):
    if action == 'right': # If we are moving right
        if state == NUM_STATES - 2:   # If we got to the right end
            next_state = 'terminal'
            reward = 1
        else:
            next_state = state + 1
            reward = 0
    else:   # If we are moving left
        reward = 0
        if state == 0: # If I'm already at the beggining
            next_state = state
        else: # If I can move left
            next_state = state - 1
    return next_state, reward

# Upadte environment to display current state
def update_env(state, episode, step_counter):
    env_list = ['x']*(NUM_STATES-1) + ['F']   # 'xxxxxxxxF' our environment
    if state == 'terminal': # If we reached the right extreme finish game
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'O'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(TIME_STEP)

# This is the main loop that will play the game
def rl():
    q_table = build_q_table(NUM_STATES, ACTIONS) # Start by initiating the Q table
    for episode in range(MAX_EPISODES): # Let's play MAX_EPISODES amount of games
        step_counter = 0 # Initialize number of steps counter
        state = 0 # Initialize agent on left corner of the game
        is_terminated = False # Define that the game is not yet terminated
        update_env(state, episode, step_counter) # Initialize the environment
        # While the game is still on
        while not is_terminated:
            action = choose_action(state, q_table) # Choose action
            q_value = q_table.loc[state, action] # Get the Q value of the current state action pair
            next_state, reward = get_env_feedback(state, action)  # Take action and get environment feedback
            if next_state != 'terminal': # Next state is not terminal
                q_update = reward + GAMMA * q_table.iloc[next_state, :].max() # Get Q value of future state and maximum action
            else: # Next state is terminal
                q_update = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_update - q_value)  # Update specific state/action Q value
            state = next_state # Move to next state
            update_env(state, episode, step_counter+1) # Update environment
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)