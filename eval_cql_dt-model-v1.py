"""
Evaluate Bi-directional Decision Transformer
"""

from sqlite3 import TimestampFromTicks
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import os, pickle
import gym, myenvs
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import mingpt.utils as utils

"""
Test 100 episodes and compute the average total reward for variouls reward-to-go values
"""
def eval_model(env, model, RtoGo_list, dtf=True, ql=0, max_time_steps=100, num_episodes=20, gamma=1.0, verbose=True):
    averaged_r = []
    pct_val = 5
    pct_l = []
    pct_h = []
    for RtoGo in RtoGo_list:
        if verbose:
            print(f'RtoGo={RtoGo}')
        r_ = []
        for n in range(num_episodes):
            if dtf:
                r = utils.play_dt(env, model, RtoGo=RtoGo, ql=ql, max_time_steps=max_time_steps, gamma=gamma)
            else: # DQN model
                r, log = model.play(env, max_steps=max_time_steps, replay=False)
            r_.append(r)
        averaged_r.append(np.mean(r_))
        pct_l.append(np.percentile(r_, pct_val))
        pct_h.append(np.percentile(r_, 100-pct_val))
        if verbose:
            print(f'reward mean: {averaged_r[-1]}, pct: {pct_l[-1]} - {pct_h[-1]}')
    results = {'RtoGo': RtoGo_list,
               'ave': averaged_r,
               'pct_l': pct_l,
               'pct_h': pct_h}
    return results

def plot_results_epochs(epochs, results, title, showfig=True, timestamp=None):
    fig, ax = plt.subplots(1)
    r_list = [r['ave'][-1] for r in results]
    ax.plot(epochs, r_list, label=f'max r: {np.max(r_list)}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Averaged total reward')
    ax.set_title(title)
    ax.legend()
    if showfig:
        fig.show()
    else:
        if timestamp is not None:
            title = title + '_' + timestamp
        fig.savefig(os.path.join('result', title + '.png'))
    return fig


""" ------------------------------------------------------------------------------------
Main
-------------------------------------------------------------------------------------"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train-dt-v1')
    parser.add_argument('--envID', type=str, default='2d_maze-v11', help='Environment name [str]') # '1d_maze-v0', cliff_walk-v0', 'pac_man-v0', '2d_maze-v0'
    parser.add_argument('--max_epochs_ql', type=int, default=100, help='max. number of steps in an episode') # 1000, 50 for 2d-maze
    parser.add_argument('--max_epochs_dt', type=int, default=200, help='max. number of steps in an episode') # 1000, 50 for 2d-maze
    parser.add_argument('--post_fix', type=str, default='_100', help='post fix for the data files [str]') # '_100' '_1000'
    parser.add_argument('--nHidden', type=int, default=32, help='Number of hidden layer units [int]')
    parser.add_argument('--nLayers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--model', type=str, default='DT', help='Select model type (DT or RvS')
    parser.add_argument('--qdt_gamma', type=float, default=1.0, help='discount factor for QDT')
    config = parser.parse_args()

    # prepare the model
    if config.model == 'DT':
        from mingpt.dt_model_v1 import GPT, DTF_2d_maze_v1x_Config
        envArgs = {}
        block_size = 6 # 6
        from mingpt.dt_model_v1 import DTF_2d_maze_v1x_Config as DTF_config
    else:
        from mingpt.RvS_model_v1 import RvS
        from mingpt.RvS_model_v1 import RvSBaseConfig as RvS_config
        envArgs = {}
        block_size = 6
        max_epochs = 1

    RtoGo_list = [50]
    
    # environment
    env = gym.make(config.envID, **envArgs)
    nActions = env.action_space.n     # number of actions
    nStates = env.observation_space.n # number of states

    # DQN model
    # from dqn.dqn_discrete_state import EnsembleDQNAgent
    # dqn_agent = EnsembleDQNAgent(nModels=5, 
    #                              nStates=nStates, 
    #                              nActions=nActions, 
    #                              lr=5e-4,                 # not used 
    #                              target_update_rate=0.01, # not used 
    #                              batch_size=128,          # not used
    #                              nEpoc=32,                # not used
    #                              max_mem=100000,
    #                              gamma=0.99, 
    #                              nHidden=[config.nHidden] * config.nLayers, 
    #                              epsilon=0.0, )
    from dqn.cqn_discrete_state import CQNAgent
    dqn_agent = CQNAgent(nModels=1, 
                         nStates=nStates, 
                         nActions=nActions, 
                         lr=5e-4,                 # not used 
                         target_update_rate=0.01, # not used 
                         batch_size=128,          # not used
                         nEpoc=32,                # not used
                         max_mem=100000,
                         gamma=0.99, 
                         nHidden=[config.nHidden] * config.nLayers, 
                         epsilon=0.0, )

    # TF model
    if config.model == 'DT':
        mconf = DTF_config(nActions, nStates, block_size)
    else:
        mconf = RvS_config(nActions, nStates, block_size)
    trj_data_file = config.envID + config.post_fix
    model_file_ql = f'dqn-b{block_size}-model-v1_' + config.envID + config.post_fix
    if config.model == 'DT':
        model_file    = f'dtf-b{block_size}-model-v1_' + config.envID + config.post_fix
    else:
        model_file    = f'RvS-b{block_size}-model-v1_' + config.envID + config.post_fix
    
    print(f"envID = {config.envID}")
    print(f"max_epochs_ql = {config.max_epochs_ql}")
    print(f"max_epochs_dt = {config.max_epochs_dt}")
    print(f"post_fix = {config.post_fix}")
    print(f"trj_data_file = {trj_data_file}")
    print(f"model_file_ql = {model_file_ql}")
    print(f"model_file = {model_file}")
    print(f"qdt_gamma = {config.qdt_gamma}")

    # evaluate 
    interval = 1 if config.max_epochs_ql < 60 else 100
    res_dqn_all_epoch = []
    epochs_dqn = []
    for e in tqdm(np.arange(0, config.max_epochs_ql, interval)):
        dqn_agent.load(os.path.join('model',model_file_ql + f'_epoch{e}.mdl'))
        res_dqn_all_epoch.append(eval_model(env, dqn_agent, RtoGo_list, dtf=False, verbose=False, num_episodes=50, gamma=config.qdt_gamma, ql=0))
        epochs_dqn.append(e)
    dqn_agent.load(os.path.join('model',model_file_ql + f'.mdl'))
    res_dqn_all_epoch.append(eval_model(env, dqn_agent, RtoGo_list, dtf=False, verbose=False, num_episodes=50, gamma=config.qdt_gamma, ql=0))
    epochs_dqn.append(config.max_epochs_ql)
    
    interval = 1 if config.max_epochs_dt < 60 else 20
    res_udrl_all_epoch = []
    epochs_udrl = []
    for e in tqdm(np.arange(0, config.max_epochs_dt, interval)):
        if config.model == 'DT':
            model = GPT(mconf)
        else:
            model = RvS(mconf)
        model.load_state_dict(torch.load(os.path.join('model',model_file + f'_epoch{e}.mdl')))
        res_udrl_all_epoch.append(eval_model(env, model, RtoGo_list, verbose=False, num_episodes=50, gamma=config.qdt_gamma, ql=0))
        epochs_udrl.append(e)
    if config.model == 'DT':
        model = GPT(mconf)
    else:
        model = RvS(mconf)
    model.load_state_dict(torch.load(os.path.join('model',model_file + '.mdl')))
    res_udrl_all_epoch.append(eval_model(env, model, RtoGo_list, verbose=False, num_episodes=50, gamma=config.qdt_gamma, ql=0))
    epochs_udrl.append(config.max_epochs_dt)

    # plot / store results
    now = datetime.now()
    timestamp = f"{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}{now.second:02d}"

    plot_results_epochs(epochs_udrl, res_udrl_all_epoch, title=model_file, showfig=False, timestamp=timestamp)
    plot_results_epochs(epochs_dqn, res_dqn_all_epoch, title=model_file_ql, showfig=False, timestamp=timestamp)
    
    with open(os.path.join('result', model_file + '_' + timestamp + '.pkl'), 'wb') as fid:
        pickle.dump([epochs_udrl, res_udrl_all_epoch, epochs_dqn, res_dqn_all_epoch], fid)
