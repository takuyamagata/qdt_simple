"""
Learn & test with CQL + Decision Transformer
"""

import numpy as np
import math
import torch
from torch.utils.data import Dataset
import os, pickle
import gym, myenvs
import argparse

class DQN_Dataset(Dataset):
    # define static variables (indexes)
    a_idx = 0   # action index
    s_idx = 1   # state index
    R_idx = 2   # reward-to-go index
    d_idx = 3   # done index
    r_idx = 4   # reward index
    ts_idx = 5  # ts index

    def __init__(self, data, num_actions, num_states):
        """
        data: trajectory data vector
        num_actions: number of actions (only used for indicating start of an episode)
        num_states: number of states (only used for BCQ case)
        """

        # store flags
        self.num_actions = num_actions
        self.num_states  = num_states

        # reshape data with the given block_size
        X = []

        # append two columns for timestep
        data = np.concatenate((data, np.zeros((len(data),2))), axis=1)

        # replace 1st action in the episode with num_actions
        ts = 1
        data[0,0] = num_actions # crude way to indicating start of episode. 
        for n in range(1,len(data)):
            if data[n-1][self.d_idx] and (not data[n][self.d_idx]):
                data[n][self.a_idx] = num_actions # crude way to indicating start of episode.
                ts = 0
            
            data[n, self.ts_idx] = ts # set timestep
            ts = ts + 1

        # convert RtoGo back to reward & append previous state
        r = np.zeros(len(data))
        for n in range(len(data)):
            if data[n, self.d_idx]: # if done == True:
                r[n] = data[n, self.R_idx]
            else:
                r[n] =  data[n, self.R_idx] - data[n+1, self.R_idx]
        data[:, self.r_idx] = r

        # re-format for DQN replay buffer (s, a, r, s', d)
        for n in range(len(data)-1):
            if not data[n, self.d_idx]: # skip done==True data
                x_ = [data[n, self.s_idx], data[n+1, self.a_idx], data[n, self.r_idx], 
                    data[n+1, self.s_idx], data[n+1, self.d_idx], ] # (s, a, r, s', done)
                X.append(x_)
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.long)
        return x

class DTF_Dataset(Dataset):
    # define static variables (indexes)
    a_idx = 0   # action index
    s_idx = 1   # state index
    R_idx = 2   # reward-to-go index
    d_idx = 3   # done index
    r_idx = 4   # reward index
    ts_idx = 5  # ts index

    def __init__(self, data, block_size, model, num_actions, num_states, gamma=1.0, replace_r=False):
        """
        data: trajectory data vector
        block_size: DTF block size (must be divisible by 3)
        model: BDTF model
        num_actions: number of actions (only used for indicating start of an episode)
        num_states: number of states (only used for BCQ case)
        replace_r: flag for replace reward-to-go with Q(s,a)
        ql_target: flag for generating Q-Learning target by using Bellman equation
        bcq: flag for Batch-Constraint-QLearning (Q-Learning target generated within the batch data) 
        """

        assert block_size % 3 == 0 # make sure block_size divisible by 3

        # store flags
        self.replace_r = replace_r # flag for replacing reward-to-go with Q(s,a)
        self.num_actions = num_actions
        self.num_states  = num_states
        self.block_size = block_size
        self.gamma = gamma
        self.data = data

        # store the model (used to generate targe Q with Bellman equation)
        self.model = model

        self.process_data(data) # prepare block input/output data (X,Y)

    def process_data(self, data):
        # append two columns for timestep
        data = np.concatenate((data, np.zeros((len(data),2))), axis=1)

        # replace 1st action in the episode with num_actions
        ts = 1
        data[0,0] = self.num_actions # crude way to indicating start of episode. 
        for n in range(1,len(data)):
            if data[n-1][self.d_idx] and (not data[n][self.d_idx]):
                data[n][self.a_idx] = self.num_actions # crude way to indicating start of episode.
                ts = 0
            
            data[n, self.ts_idx] = ts # set timestep
            ts = ts + 1

        # convert RtoGo back to reward & append previous state
        r = np.zeros(len(data))
        for n in range(len(data)):
            if data[n, self.d_idx]: # if done == True:
                r[n] = data[n, self.R_idx]
            else:
                r[n] =  data[n, self.R_idx] - data[n+1, self.R_idx]
        data[:, self.r_idx] = r

        # reshape data with the given block_size
        print(f"replace flag:{self.replace_r}")
        if self.replace_r == 1:
            print('Replacing reward enabled')
        else:
            print('No replacing reward!!')
        X, Y = [], []
        for n in np.flip( np.arange(len(data) - block_size//3) ):
            if np.any(data[n:n+block_size//3, 3]):
                continue
            x_ = data[n:n+block_size//3]
            y_ = data[n+1:n+1+block_size//3]

            if self.replace_r == 1:
                if y_[-1, self.d_idx]: # if done?
                    x_[-1, self.R_idx] = x_[-1, self.r_idx]
                else:
                    q_ = self.model.get_q(y_[-1, self.s_idx])
                    q_ = torch.max(q_, dim=1).values
                    
                    if y_[-1, self.R_idx] < q_:
                        x_[-1, self.R_idx] = self.gamma * q_ + x_[-1, self.r_idx]
                    else:
                        x_[-1, self.R_idx] = self.gamma * y_[-1, self.R_idx] + x_[-1, self.r_idx]
                
                consistency_relabel = True
                if consistency_relabel:
                    # with consistency relabel ###################################################
                    for k in range(self.block_size//3-1):
                        x_[-k-2, self.R_idx] = self.gamma * x_[-k-1, self.R_idx] + x_[-k-2, self.r_idx]
                else:                
                    # without consistency relabel ################################################              
                    for k in range(self.block_size//3-1):
                        q_ = self.model.get_q(y_[-k-2, self.s_idx])
                        q_ = torch.max(q_, dim=1).values
                        if x_[-k-1, self.R_idx] < q_:
                            x_[-k-2, self.R_idx] = self.gamma * q_ + x_[-k-2, self.r_idx]
                        else:
                            x_[-k-2, self.R_idx] = self.gamma * x_[-k-1, self.R_idx] + x_[-k-2, self.r_idx]
                    ##############################################################################


            X.append(x_.copy())
            Y.append(y_.copy())

        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.long)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y

""" ------------------------------------------------------------------------------------
Main
-------------------------------------------------------------------------------------"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train-dt-v1')
    parser.add_argument('--envID', type=str, default='2d_maze-v11', help='Environment name [str]') # '1d_maze-v0', cliff_walk-v0', 'pac_man-v0', '2d_maze-v11'
    parser.add_argument('--max_epochs_ql', type=int, default=1000, help='max. number of steps in an episode') # 1000, 50 for 2d-maze
    parser.add_argument('--max_epochs_dt', type=int, default=2000, help='max. number of steps in an episode') # 1000, 50 for 2d-maze
    parser.add_argument('--post_fix', type=str, default='_100', help='post fix for the data files [str]') # '_100' '_1000'
    
    parser.add_argument('--nHidden', type=int, default=32, help='Number of hidden layer units [int]')
    parser.add_argument('--nLayers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--target-update-rate', type=float, default=1e-2, help='Target DQN weights update rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--max-mem', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--nEpoc', type=int, default=None, help='Number of epoch for training')
    parser.add_argument('--model', type=str, default='DT', help='select RL agent model (DT or RvS)')
    parser.add_argument('--qdt_gamma', type=float, default=1.0, help='discount factor for QDT')
    parser.add_argument('--cql_gamma', type=float, default=0.99, help='discount factor for CQL')
    parser.add_argument('--replace', type=int, default=1)
    config = parser.parse_args()

    # prepare the model
    if config.model == 'DT':
        print('Use DT model')
        from mingpt.dt_model_v1 import GPT, DTF_2d_maze_v1x_Config
        envArgs = {}
        block_size = 6 # 6
        max_epochs = 1
        from mingpt.dt_model_v1 import DTF_2d_maze_v1x_Config as DTF_config
    else:
        print('Use RvS model')
        from mingpt.RvS_model_v1 import RvS
        from mingpt.RvS_model_v1 import RvSBaseConfig as RvS_config
        envArgs = {}
        block_size = 6
        max_epochs = 1
    
    # environment
    env = gym.make(config.envID, **envArgs)
    nActions = env.action_space.n     # number of actions
    nStates = env.observation_space.n # number of states

    model_file_ql = f'dqn-b{block_size}-model-v1_' + config.envID + config.post_fix
    if config.model == 'DT':
        mconf = DTF_config(nActions, nStates, block_size)
        model = GPT(mconf)
        model_file = f'dtf-b{block_size}-model-v1_' + config.envID + config.post_fix
    # TF model
    else:
        mconf = RvS_config(nActions, nStates, block_size)
        model = RvS(mconf)
        model_file = f'RvS-b{block_size}-model-v1_' + config.envID + config.post_fix

    trj_data_file = config.envID + config.post_fix

    print(f"envID = {config.envID}")
    print(f"max_epochs_ql = {config.max_epochs_ql}")
    print(f"max_epochs_dt = {config.max_epochs_dt}")
    print(f"post_fix = {config.post_fix}")
    print(f"replace = {config.replace}")
    print(f"trj_data_file = {trj_data_file}")
    print(f"model_file = {model_file}")
    print(f"CQL Gamma = {config.cql_gamma}")
    print(f"QDT Gamma = {config.qdt_gamma}")

    # ----- Q-Learning stage ----------------
    # from dqn.dqn_discrete_state import EnsembleDQNAgent
    # dqn_agent = EnsembleDQNAgent(nModels=5, 
    #                              nStates=nStates, 
    #                              nActions=nActions, 
    #                              lr=config.lr, 
    #                              target_update_rate=config.target_update_rate, 
    #                              batch_size=config.batch_size, 
    #                              nEpoc=None,
    #                              max_mem=config.max_mem,
    #                              gamma=0.99, 
    #                              nHidden=[config.nHidden] * config.nLayers, 
    #                              epsilon=0.0, )
    from dqn.cqn_discrete_state import CQNAgent
    dqn_agent = CQNAgent(nModels=1, 
                         nStates=nStates, 
                         nActions=nActions, 
                         lr=config.lr, 
                         target_update_rate=config.target_update_rate, 
                         batch_size=config.batch_size, 
                         nEpoc=None,
                         max_mem=config.max_mem,
                         gamma=config.cql_gamma, 
                         nHidden=[config.nHidden] * config.nLayers, 
                         epsilon=0.0, )

    # prepare training data
    with open(os.path.join('data', trj_data_file + '.pkl'), "rb") as fid:
        data = pickle.load( fid )
    train_dataset = DQN_Dataset(data, 
                                num_actions=nActions, 
                                num_states=nStates, )
    
    # upload data into experience-replay-buffer
    for n in range(len(train_dataset)):
        dqn_agent.observe(train_dataset[n]) # (s, a, r, s_, done)

    # train ensemble-DQN
    skipDQN_training = False
    if skipDQN_training == False:
        interval = 1 if config.max_epochs_ql < 60 else 100
        for e in range(config.max_epochs_ql):
            dqn_agent.replay()
            dqn_agent.update_target_DQN()
            if e % interval==0:
                dqn_agent.save(os.path.join('model', model_file_ql + f'_epoch{e}.mdl')) # store model
                # plot learned Q (only works for 2d_maze envirionment)
                # print(f"----- epoch: {e}")
                # for s in range(6):
                #     print(f"state:{s*7}")
                #     q = dqn_agent.get_q(s*7)    
                #     print(f"mean: {torch.mean(q,dim=0).data}")
                #     print(f"std : {torch.std(q, dim=0).data}")
                #     q = dqn_agent.get_q(s*7, target=True)    
                #     print(f"mean: {torch.mean(q,dim=0).data}")
                #     print(f"std : {torch.std(q, dim=0).data}")

        # # store the final model for the 1st stage (Q-Learning stage)
        dqn_agent.save(os.path.join('model', model_file_ql + '.mdl'))
    else:
        dqn_agent.load(os.path.join('model', model_file_ql + '_epoch1000.mdl'))
       
    # ----- UDRL learning stage ----------------
    train_dataset = DTF_Dataset(data, 
                                block_size=block_size, 
                                model=dqn_agent, 
                                num_actions=nActions, 
                                num_states=nStates,
                                gamma=config.qdt_gamma,
                                replace_r=config.replace)   #### <=== False for Replace OFF!!!
    from mingpt.trainer import TrainerBDTF, TrainerConfig
    # initialize a trainer instance and kick off training
    if config.model == 'DT':
        tconf = TrainerConfig(max_epochs=config.max_epochs_dt, batch_size=64, learning_rate=4e-4, # 6e-5
                            lr_decay=True, warmup_tokens=32*10, final_tokens=2*len(train_dataset)*block_size,
                            num_workers=2)
    else:
        tconf = TrainerConfig(max_epochs=config.max_epochs_dt, batch_size=64, learning_rate=2e-4, # 6e-5
                    lr_decay=False, warmup_tokens=32*10, final_tokens=2*len(train_dataset)*block_size,
                    num_workers=2)


    trainer = TrainerBDTF(model, train_dataset, None, tconf)
    trainer.train(store_model=os.path.join('model', model_file), ql=0)

    # store the learned model
    torch.save(model.state_dict(), os.path.join('model', model_file + '.mdl'))
