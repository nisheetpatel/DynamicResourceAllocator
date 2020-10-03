import numpy as np
import pandas as pd

"""
Defines the class environment as per Huys-Dayan-Rosier's planning task
& the agent's long-term tabular memories: (s,a), r(s,a), Q(s,a), pi(a|s).
"""
class HuysTask:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(self, depth=3, n_states=6):
        self.depth = depth
        transitions = np.mat('0 1 0 1 0 0;\
                            0 0 1 0 1 0;\
                            0 0 0 1 0 1;\
                            0 1 0 0 1 0;\
                            1 0 0 0 0 1;\
                            1 0 1 0 0 0')
        rewards = np.mat('0   140  0   20  0   0; \
                          0   0   -20  0  -70  0; \
                          0   0    0  -20  0  -70;\
                          0   20   0   0  -20  0; \
                         -70  0    0   0   0  -20;\
                         -20  0    20  0   0   0')

        # Setting up the transitions and rewards matrices for the
        # extended state space: 6 -> 6 x T_left
        self.transition_matrix = np.zeros(((depth+1)*n_states,(depth+1)*n_states),dtype=int)
        self.reward_matrix = np.zeros(((depth+1)*n_states,(depth+1)*n_states),dtype=int)

        nrows = transitions.shape[0]
        Nrows = self.transition_matrix.shape[0]

        for i in range(nrows,Nrows,nrows):
            self.transition_matrix[i-nrows:i,i:i+nrows] = transitions

        for i in range(nrows,Nrows,nrows):
            self.reward_matrix[i-nrows:i,i:i+nrows] = rewards

        # Transitions, rewards, states
        self.n_states = len(self.transition_matrix)
        self.states = np.arange(self.n_states)
        self.transitions = np.transpose(self.transition_matrix.nonzero())
        self.rewards = np.array(self.reward_matrix[self.reward_matrix != 0])


"""
Custom-made two-step T-maze with 14 states.
"""
class Tmaze:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(self, depth=3, n_states=6, gridworld=False):
        self.depth = depth
        self.gridworld = gridworld
        self.transition_matrix = \
            np.mat('0 1 0 0 0 0 0 0 0 0 0 0 0 0;\
                    1 0 1 1 0 0 0 0 0 0 0 0 0 0;\
                    0 1 0 0 1 0 0 0 0 0 0 0 0 0;\
                    0 1 0 0 0 1 0 0 0 0 0 0 0 0;\
                    0 0 1 0 0 0 1 0 0 0 0 0 0 0;\
                    0 0 0 1 0 0 0 1 0 0 0 0 0 0;\
                    0 0 0 0 1 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 1 0 0 0 1 0 0 0 0;\
                    0 0 0 0 0 0 1 0 0 0 1 0 1 0;\
                    0 0 0 0 0 0 0 1 0 0 0 1 0 1;\
                    0 0 0 0 0 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 0 0 0 0 1 0 0 0 0;\
                    0 0 0 0 0 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 0 0 0 0 1 0 0 0 0')
        self.reward_matrix       = -self.transition_matrix
        self.reward_matrix[8,10] =  10
        self.reward_matrix[8,12] =  -5
        self.reward_matrix[9,11] = -10
        self.reward_matrix[9,13] =  5

        # Transitions, rewards, states
        self.n_states = len(self.transition_matrix)
        self.states = np.arange(self.n_states)
        self.transitions = np.transpose(self.transition_matrix.nonzero())
        self.rewards = np.array(self.reward_matrix[self.reward_matrix != 0])


""" 
A wrapper class for a maze, containing all the information about the maze.
Initialized to the 2D maze used by Mattar & Daw 2019 by default, however, 
it can be easily adapted to any other maze by redefining obstacles and size.
"""
class Maze:
    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [2, 0]

        # goal state
        self.GOAL_STATES = [[0, 8]]
        self.new_goal = [[0, 6]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = [[0,7]]
        self.new_obstacles = None

        # time to change environment
        self.switch_time = 3000

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

    # take @action in @state
    # @return: [new state, reward]
    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 10
        else:
            reward = -1
        return [x, y], reward
    
    # Swith it up
    def switch(self):
        self.obstacles = [x for x in self.obstacles if x not in self.old_obstacles]
        self.GOAL_STATES = self.new_goal
        return