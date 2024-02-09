import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


def play(env, model, RtoGo=100, render=False, max_time_steps=10000):
    """ Run one episode on a given environment """    
    trj_history = []
    
    s, r, done, info = env.reset()
    a = env.action_space.n # initial action indicator
    ts = 0

    r_total = r    
    while True:
        # append data to the trajectory buffer
        trj_history.append([a,s,RtoGo - r_total, done, ts])
        if render: print(f"state:{s}, action:{a}, reward:{r}")
        # sample action
        inp = torch.tensor(np.array(trj_history))[-model.block_size_in_ts::,:].unsqueeze(0)
        logits, _ = model(inp)
        a = torch.argmax(logits[0,-1,:env.action_space.n]).numpy()
        # step
        s, r, done, info = env.step(a)
        r_total += r
        ts += 1
        if done or (ts > max_time_steps): break
    return r_total

def play_dt(env, model, RtoGo=100, gamma=1.0, render=False, max_time_steps=10000, ql=False):
    """ Run one episode on a given environment """    
    trj_history = []
    
    s, r, done, info = env.reset()
    a = env.action_space.n # initial action indicator
    ts = 0

    r_total = r
    RtoGo_ = RtoGo 
    while True:
        # compute RtoGo
        if RtoGo is not None:
            RtoGo_ = RtoGo_/gamma - r
        else:
            if len(trj_history) == 0: # if trj_history is empty
                inp = torch.tensor([[[a,s,100 - r_total, done, ts]]])
            else:
                inp = torch.tensor(np.array(trj_history))[-model.block_size_in_ts::,:].unsqueeze(0)
            _, q, _ = model(inp, ql=1)
            RtoGo_ = torch.max(q[0,-1,:env.action_space.n]).detach().numpy() #.astype(np.float16)
            
        # append data to the trajectory buffer
        trj_history.append([a,s,RtoGo_, done, ts])
        # sample action
        inp = torch.tensor(np.array(trj_history))[-model.block_size_in_ts::,:].unsqueeze(0)
        if render: print(f"state:{s}, action:{a}, reward:{r}")
        if ql:
            _, logits, _ = model(inp, ql=1)
        else:
            logits, _, _ = model(inp, ql=0)
        if logits.dim() == 3:
            a = torch.argmax(logits[0,-1,:env.action_space.n]).numpy()
        else:
            a = torch.argmax(logits[0,:env.action_space.n]).numpy()
        # step
        s, r, done, info = env.step(a)
        r_total += r
        ts += 1
        if done or (ts > max_time_steps): break
    return r_total
