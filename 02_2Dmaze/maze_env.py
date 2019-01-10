"""
This is a simple maze example for reinforcement learning
There is an agent on a 2D maze that needs to get to the finish square
Blue rectangle: agent
Black rectangles: penalty states (reward = -1]
Yellow rectangle: Finish [reward = 1]
All other rectangles: nothing [reward = 0]
"""

import numpy as np
import time
import tkinter as tk

# Define hyperparameters
UNIT = 40 # Size of the pixels
MAZE_X = 6 # Width of the maze
MAZE_Y = 6 # Height of the maze
HOLE1_LOC_X = 2 # Hole 1 location in X
HOLE1_LOC_Y = 3 # Hole 1 location in Y
HOLE2_LOC_X = 4 # Hole 2 location in X
HOLE2_LOC_Y = 4 # Hole 2 location in Y
TARGET_LOC_X = 3 # Target location in X
TARGET_LOC_Y = 5 # Target location in Y

# Define the Maze class as inherited from Tkinter
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right'] # Define possible actions
        self.n_actions = len(self.action_space) # Count number of actions available
        self.geometry('{0}x{1}'.format(MAZE_Y * UNIT, MAZE_Y * UNIT)) # Define the geometry
        self._build_maze() # Call build maze function

    # This function creates the maze using the tkinter canvas
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_Y * UNIT,
                           width=MAZE_X * UNIT)
        for c in range(0, MAZE_X * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_Y * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_Y * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_X * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])

        hell1_center = origin + np.array([UNIT * HOLE1_LOC_X, UNIT * HOLE1_LOC_Y])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')

        hell2_center = origin + np.array([UNIT * HOLE2_LOC_X, UNIT * HOLE2_LOC_Y])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        target_center = origin + np.array([UNIT * TARGET_LOC_X, UNIT * TARGET_LOC_Y])
        self.target = self.canvas.create_rectangle(
            target_center[0] - 15, target_center[1] - 15,
            target_center[0] + 15, target_center[1] + 15,
            fill='yellow')

        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='blue')

        self.canvas.pack()

    # This function resets the maze and replaces the blue rectangle at origin
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='blue')
        return self.canvas.coords(self.rect)

    # Re-render the maze when taking an action
    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_Y - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_X - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # Move rectangle based on action
        self.canvas.move(self.rect, base_action[0], base_action[1])

        # Get new state of rectangle as his coordinates
        s_ = self.canvas.coords(self.rect)

        # reward function
        if s_ == self.canvas.coords(self.target): # If we found the target
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]: # If we found a hole
            reward = -1
            done = True
            s_ = 'terminal'
        else: # Other
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()