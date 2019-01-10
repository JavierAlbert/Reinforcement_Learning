"""
This is a simple maze example for reinforcement learning
There is an agent on a 2D maze that needs to get to the finish square
Blue rectangle: agent
Black rectangles: penalty states (reward = -1]
Yellow rectangle: Finish [reward = 1]
All other rectangles: nothing [reward = 0]
"""

from maze_env import Maze
from learning import QLearning

NUM_GAMES = 20

def update():
    for episode in range(NUM_GAMES):
        observation = env.reset() # Initialize observation

        while True:
            env.render() # Render environment
            action = RL.choose_action(str(observation)) # Choose action based on current state
            observation_, reward, done = env.step(action) # Take step and get feedback from environment
            RL.learn(str(observation), action, reward, str(observation_)) # Learn from the step
            observation = observation_ # Make new observation the current state
            if done:
                break
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze() # Initialize Maze environment
    RL = QLearning(actions=list(range(env.n_actions))) # Initialize QLearning object
    env.after(100, update)
    env.mainloop()