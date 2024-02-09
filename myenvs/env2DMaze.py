"""
2D Maze environment
    N x N grid world environment.
    Start from bottom-left corner, and goal in top-right corner.
    Reward: +10 @ gloal, -1 @ each step
    Action: 8 actions -> move S/N/E/W/SW/SE/NW/NE
"""

import numpy as np
import gym
from gym import error, spaces, utils
import torch

class Env(gym.Env):
    def __init__(self, N=6, R=[+10, -1]):
        self.R = R # rewards
        self.N = N # maze size
        self.move = [[ 0,+1], # 0 up
                     [+1,+1], # 1 up-right
                     [+1, 0], # 2 right
                     [+1,-1], # 3 down-right
                     [ 0,-1], # 4 down
                     [-1,-1], # 5 down-left
                     [-1, 0], # 6 left
                     [-1,+1],]# 7 up-left
        self.reset()
        return
    
    def reset(self):
        self.x    = 0
        self.y    = 0
        return self.state, 0, False, []
    
    def step(self, action):
        # init. return parameters
        rw = 0
        done = False
        info = []

        # Check if in absorbing state
        if self.state == self.N**2 - 1:
            done = True
            return self.state, rw, done, info
        
        # State change
        x_ = self.x + self.move[action][0]
        self.x = x_ if x_ < self.N and x_ >= 0 else self.x
        y_ = self.y + self.move[action][1]
        self.y = y_ if y_ < self.N and y_ >= 0 else self.y

        rw = self.R[1] # reward for every step
        if self.state == self.N**2-1: # Goal !!
            rw += self.R[0]
            done = True
        
        return self.state, rw, done, info

    def render(self, mode='Human'):
        img = []
        return img
    
    @property
    def state(self):
        return self.x + self.y * self.N
    
    @property
    def action_space(self):
        return spaces.Discrete(8) # number of actions (8)

    @property
    def observation_space(self):
        return spaces.Discrete(self.N**2) # number of states (N**2)

    def optimalQ(self, s,a):
        if s == self.N**2 - 1:
            value = 0
        else:
            # get x,y
            x = s % self.N
            y = torch.div(s-x, self.N, rounding_mode='trunc')
            # move by a
            x_ = x + self.move[a][0]
            x = x_ if x_ < self.N and x_ >= 0 else x
            y_ = y + self.move[a][1]
            y = y_ if y_ < self.N and y_ >= 0 else y
            # get value from (x,y)
            value = self.R[0] + np.max([self.N-x-1, self.N-y-1]) * self.R[1] + self.R[1]
        return value

    def optimalV(self, s):
        if s == self.N**2 - 1:
            value = 0
        else:
            # get x,y
            x = s % self.N
            y = torch.div(s-x, self.N, rounding_mode='trunc')
            # get value from (x,y)
            value = self.R[0] + np.max([self.N-x-1, self.N-y-1]) * self.R[1]
        return value


class Env_v1(Env):
    def __init__(self, N=6, R=[+10, -1]):
        self.R = R # rewards
        self.N = N # maze size
        self.move = [[ 0,+1], # 0 up
                     [+1,+1], # 1 up-right
                     [+1, 0], # 2 right
                     [+1,-1], # 3 down-right
                     [ 0,-1], # 4 down
                     [-1,-1], # 5 down-left
                     [-1, 0], # 6 left
                     [-1,+1],]# 7 up-left

        # generate random action mapping for each state
        self.action_mapping = np.zeros((self.observation_space.n, self.action_space.n))
        for s in range(self.observation_space.n):
            self.action_mapping[s,:] = np.roll(np.arange(self.action_space.n), s) # np.random.permutation(self.action_space.n)
        self.action_mapping = self.action_mapping.astype(int)
        self.reset()
        return

    def step(self, action):
        # init. return parameters
        rw = 0
        done = False
        info = []

        # Check if in absorbing state
        if self.state == self.N**2 - 1:
            done = True
            return self.state, rw, done, info
        
        # State change
        action_ = self.action_mapping[self.state, action] # re-map action
        x_ = self.x + self.move[action_][0]
        self.x = x_ if x_ < self.N and x_ >= 0 else self.x
        y_ = self.y + self.move[action_][1]
        self.y = y_ if y_ < self.N and y_ >= 0 else self.y

        rw = self.R[1] # reward for every step
        if self.state == self.N**2-1: # Goal !!
            rw += self.R[0]
            done = True
        
        return self.state, rw, done, info

    def optimalQ(self, s,a):
        # done?
        if s == self.N**2 - 1:
            value = 0
        else:
            # get x,y
            x = s % self.N
            y = torch.div(s-x, self.N, rounding_mode='trunc')
            # move by a
            a = self.action_mapping[s, a] # re-map action
            x_ = x + self.move[a][0]
            x = x_ if x_ < self.N and x_ >= 0 else x
            y_ = y + self.move[a][1]
            y = y_ if y_ < self.N and y_ >= 0 else y
            # get value from (x,y)
            value = self.R[0] + np.max([self.N-x-1, self.N-y-1]) * self.R[1] + self.R[1]
        return value

""" R=+100 for goal, R=-10 for each step """
class Env_v10(Env):
    def __init__(self, N=6, R=[+100, -10]):
        super().__init__(N, R)


class Env_v11(Env_v1):
    def __init__(self, N=6, R=[+100, -10]):
        super().__init__(N,R)


if __name__ == '__main__':

    e = Env_v11()

    a = None
    s, r, done, info = e.reset()
    for n in range(10):
        print(f"n:{n}, a:{a}, s:{s}, r:{r}, done:{done}")
        if done: break
        a = np.random.randint(3) # only takes actions for UP, UP-RIGHT or RIGHT
        s,r,done,info = e.step(a)

    print('------------')

    a = None
    s, r, done, info = e.reset()
    for n in range(20):
        print(f"n:{n}, a:{a}, s:{s}, r:{r}, done:{done}")
        if done: break
        a = e.action_space.sample()
        s,r,done,info = e.step(a)
