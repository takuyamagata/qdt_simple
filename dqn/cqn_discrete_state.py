# -*- coding: utf-8 -*-
"""
DQN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import random, pickle
import argparse

import gym, myenvs

# DQN definition
class DQNet(nn.Module):
    def __init__(self, nStates, nActions, nStates_emb, nHidden):
        super(DQNet, self).__init__()
        self.nStates = nStates
        self.nStates_emb = nStates_emb
        self.nActions = nActions
        self.nHidden = nHidden
        self.s_emb = nn.Embedding(self.nStates, self.nStates_emb)
        layers = []
        layers.append( nn.Linear(self.nStates_emb, self.nHidden[0]) ) # input layer
        for n in range(len(self.nHidden)-1):
            layers.append( nn.Linear(self.nHidden[n], self.nHidden[n+1]) ) # hidden layer(s)
        layers.append(nn.Linear(self.nHidden[-1], self.nActions) ) # output layer
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        x = self.s_emb(x)
        for n in range(len(self.layers)-1):
            x = F.relu(self.layers[n](x))
        out = self.layers[-1](x) # linear activation function at the final layer
        return out
    
# ---------------------------------------------------------------------------
# Memory - handling replay buffer
class Memory:
    samples = []    # replay buffer
    
    def __init__(self, capacity):
        self.capacity = capacity
    
    # add sample to the buffer    
    def add(self, sample):
        self.samples.append(sample)
        
        if len(self.samples) > self.capacity:
            self.samples.pop(0)
            
    # get randomly selected n samples from the buffer
    def sample(self, n):
        n = min(n, len(self.samples))
        
        return random.sample(self.samples, n)
    
# ---------------------------------------------------------------------------
# Agent - toplevel class 

class CQNAgent():
    
    def __init__(self, nModels, nStates, nActions, lr, target_update_rate, batch_size, nEpoc, max_mem,
                 gamma=1.0, nHidden=[24, 24], epsilon=0.02, min_q_weight=0.5):
        self.nModels = nModels
        self.nStates = nStates
        self.nActions = nActions
        self.nHidden = nHidden
        self.epsilon = epsilon
        self.lr = lr
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.nEpoc = nEpoc
        self.gamma = gamma
        self.min_q_weight = min_q_weight
        
        self.memory = Memory(max_mem)
        self.DQNet     = [DQNet(nStates, nActions, nHidden[0], nHidden) for _ in range(self.nModels)]
        self.DQNet_tgt = [DQNet(nStates, nActions, nHidden[0], nHidden) for _ in range(self.nModels)]
        
        self.optim = [torch.optim.Adam(self.DQNet[n].parameters(), lr=self.lr) for n in range(self.nModels)]
        
    def act(self, s, m=0):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nActions, ())
        else:
            return torch.argmax(self.DQNet[m](torch.tensor(s)))

    def act_exp(self, s): # take max across models, then take argmax across actions
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nActions, ())
        else:
            Q_ = torch.max(self.get_q(s), dim=0).values
            return torch.argmax(Q_)

    def act_safe(self, s): # take min across models, then take argmax across actions
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nActions, ())
        else:
            Q_ = torch.min(self.get_q(s), dim=0).values
            return torch.argmax(Q_)

    def get_q(self, s, target=False): # return Q matrix. each row -> each model, each column -> each action
        if target:
            q = torch.stack([self.DQNet_tgt[m](torch.tensor(s).type(torch.long)) for m in range(self.nModels)], dim=0)
        else:
            q = torch.stack([self.DQNet[m](torch.tensor(s).type(torch.long)) for m in range(self.nModels)], dim=0)
        return q

    def observe(self, sample):
        self.memory.add(sample)
        self.update_target_DQN()
        
    def update_target_DQN(self, m=None):
        if m is None:
            for m in range(self.nModels):
                for org, target in zip(self.DQNet[m].parameters(), self.DQNet_tgt[m].parameters()):
                    target.data.copy_( self.target_update_rate * org.data + (1-self.target_update_rate) * target.data )
        else:
            for org, target in zip(self.DQNet[m].parameters(), self.DQNet_tgt[m].parameters()):
                target.data.copy_( self.target_update_rate * org.data + (1-self.target_update_rate) * target.data )
    
    def replay(self, m=None):
        if m is None:
            for m in range(self.nModels):
                self.replay_(m)
        else:
            self.replay_(m)
        
    def replay_(self, m):
        batch = self.memory.sample(self.batch_size)
        batchLen = len(batch)

        if batchLen == 0:
            return

        [states, a, r, states_, done] = zip(*batch)
        states = torch.tensor(states)
        states_ = torch.tensor(states_)

        self.optim[m].zero_grad()
        p = self.DQNet[m](states).view(-1, self.nActions)
        with torch.no_grad():
            p_ = self.DQNet_tgt[m](states_).view(-1, self.nActions)
            p_n = self.DQNet[m](states_).view(-1, self.nActions)

            y = torch.zeros(batchLen, self.nActions)
            for k in range(batchLen):
                t = p[k].clone()  # target value
                if done[k]:
                    t[a[k]] = r[k]
                else:
                    t[a[k]] = r[k] + self.gamma * p_[k][torch.argmax(p_n[k])]  # Double Q-network
                    # t[a] = r + self.gamma  * torch.max(p_[n])
                    # t[a] = r + self.gamma  * torch.max(p_n[n])

                y[k] = t
        loss = torch.sum(0.5 * (y - p) ** 2) / batchLen

        # Conservative Q-Learning loss
        logsumexp = torch.sum(torch.logsumexp(p, 1)) / batchLen
        dataset_expec = torch.sum( p[np.arange(p.shape[0]), a]) / batchLen
        loss += (logsumexp - dataset_expec) * self.min_q_weight

        loss.backward()
        self.optim[m].step()
    
    # save / load model
    def save(self, fname):
        with open(fname, 'wb') as fid:
            savedata = [[self.DQNet[m].state_dict(), self.DQNet_tgt[m].state_dict()] for m in range(self.nModels)]
            savedata.append(self.memory.samples)
            pickle.dump(savedata, fid)

    def load(self, fname):
        with open(fname, 'rb') as fid:
            loaddata = pickle.load(fid)
            for m in range(self.nModels):
                self.DQNet[m].load_state_dict(loaddata[m][0])
                self.DQNet_tgt[m].load_state_dict(loaddata[m][1])
                self.memory.samples = loaddata[-1]

    # run one episode
    def play(self, env, m=None, max_steps=100, render=False, replay=True):
        
        if m is None:
            m = np.random.randint(self.nModels) # if m is None, randomly select a model
        
        s, r, done, info = env.reset()
        # s = s.astype('float32')
        logs = {"video_frames": [],
                "actions": [],
                "observations": [],
                "rewards": []}
        totRW = 0                     # total reward in this episode
        sq_error = 0  # squared distance
        done = False                  # episode completion flag
        
        for j in range(max_steps):
            # call agent
            action = self.act(s)
            a = action.data.item()

            if render:
                img = env.render()
                logs["video_frames"].append(img)
                logs["actions"].append(a)
                logs["observations"].append(s)
                logs["rewards"].append(r)

            # if done==True, then finish this episode.
            if done:
                break
                
            # call environment
            o, r, done, info = env.step(a)

            # store experience into replay buffer
            s_ = o
            self.observe( (s, a, r, s_, done) )
            s = s_

            # accumrate total reward
            totRW += r

        # learaning
        if replay:
            if self.nEpoc is None:
                nEpoc = len(self.memory.samples) // self.batch_size
            else:
                nEpoc = self.nEpoc
            for n in range(nEpoc):
                self.replay()
        
        return totRW, logs
