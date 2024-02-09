"""
Bi-directional Decision Transformer model:
- It learns both UD(Upside-Down)-RL and Q-Learning (Or SARSA) at same time.
- UD-RL learns (state, RtoGo) -> action
- Q-Learning learns (state, action) -> RtoGo
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, action_size, state_size, block_size, **kwargs):
        self.action_size = action_size
        self.state_size = state_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class DTF_2d_maze_v1x_Config(GPTConfig):
    """ Decision Transformer config for cliff_walk """
    n_layer = 4
    n_head = 4
    n_embd = 64 # 64 # n_embd must be divisible by n_head
    max_timestep = 128 # max. possible time-steps in a episode
    q_loss_weight = 0.2
    bcq = False


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.repeat_interleave( torch.repeat_interleave(torch.tril(torch.ones(config.block_size//3,config.block_size//3)),3, dim=1), 3,dim=0)
        mask_ql = mask.clone()
        mask_ql[:,1::3] = 0 # masked RtoGo keys
        # mask_ql[1::3,:] = 0 # masked RtoGo queries
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask_ql", mask_ql.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.ql_flag = False

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.ql_flag:
            att = att.masked_fill(self.mask_ql[:,:,:T,:T] == 0, float('-inf'))
        else:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.s_emb = nn.Embedding(config.state_size, config.n_embd)
        self.a_emb = nn.Embedding(config.action_size+1, config.n_embd) # +1 for indicating start of the episode
        self.R_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.seg_emb = nn.Parameter(torch.zeros(1, 3, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size//3, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # self.blocks = [Block(config) for _ in range(config.n_layer)]
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.action_size+1, bias=False)
        # self.head = nn.Sequential(nn.Linear(config.n_embd, config.n_embd*2), nn.GELU(), nn.Linear(config.n_embd*2, config.action_size+1))

        self.block_size = config.block_size
        self.block_size_in_ts = config.block_size//3 # block size in time-steps
        self.action_size = config.action_size
        self.n_embd = config.n_embd
        self.max_timestep = config.max_timestep
        self.q_loss_weight = config.q_loss_weight
        if hasattr(config, 'bcq'):
            self.bcq = config.bcq
        else:
            self.bcq = False
        if self.bcq:
            # prepare BCQ table (A,A,A,S,S,S, A) in case of block_size=6
            self.bcq_table = nn.Parameter(torch.zeros([*torch.repeat_interleave(torch.tensor([self.action_size+1, config.state_size]), self.block_size_in_ts), self.action_size+1]), requires_grad=False)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('seg_emb')
        if self.bcq:
            no_decay.add('bcq_table')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, ql=0.5):
        ql_flag = (torch.rand(1) < ql) # q-learning mode flag (change the mask not to use RtoG in self-attension layer)
        for m in self.blocks:
            m.attn.ql_flag = ql_flag

        a = idx[:,:,0].type(torch.int)
        s = idx[:,:,1].type(torch.int)
        R = idx[:,:,2]
        b_s, t_s = s.size()
        b_a, t_a = a.size()
        b_R, t_R = R.size()
        assert b_s == b_R and b_a == b_s
        assert t_s == t_R and t_s == t_a 
        assert t_s <= self.block_size, "Cannot forward, model block size is exhausted."

        # begginig of episode
        a[a >= self.action_size] = self.action_size

        # forward the GPT model
        embeddings = torch.zeros(b_s, t_s*3, self.n_embd)
        embeddings[:,0::3,:] = self.a_emb(a) + self.pos_emb[:,:t_a,:] #+ self.seg_emb[:,0,:]
        embeddings[:,1::3,:] = self.R_emb(R.type(torch.float32).unsqueeze(2)) + self.pos_emb[:,:t_R,:] #+ self.seg_emb[:,1,:]
        embeddings[:,2::3,:] = self.s_emb(s) + self.pos_emb[:,:t_s,:] #+ self.seg_emb[:,2,:]
        
        x = self.drop(embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        logits_a = logits[:,0::3,:] # keep logits for action embeddings
        logits_q = logits[:,2::3,:] # keep logits for state embeddings for Q-function

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if ql_flag:
                # loss for Q-function
                targets_q = targets[:,:,2]
                targets_a = targets[:,:, 0].long()
                targets_a[targets_a >= self.action_size] = self.action_size
                q = torch.gather(logits_q, 2, targets_a.unsqueeze(2))
                # loss = self.q_loss_weight * F.huber_loss(q.squeeze(2), targets_q) # balance losses
                loss = self.q_loss_weight * F.mse_loss(q.squeeze(2), targets_q) # balance losses
            else:
                # loss for action
                targets_a = targets[:,:, 0].long()
                targets_a[targets_a >= self.action_size] = self.action_size
                loss = F.cross_entropy(logits_a.view(-1, logits_a.size(-1)), targets_a.view(-1))

        if targets is not None:
            # update BCQ table
            if self.bcq:
                for n in range(b_s):
                    self.bcq_table[[*a[n,:], *s[n,:], targets[n,-1,0].type(torch.long)]] = True
        else:
            # BCQ (replace Q output with -inf for un-seen action)
            if self.bcq:
                for n in range(b_s):
                    # if torch.any(self.bcq_table[[*a[n,:], *s[n,:], torch.arange(self.action_size+1)]] == True):
                    if len(a) == self.block_size_in_ts:
                        logits_q[n, -1, self.bcq_table[[*a[n,:], *s[n,:], torch.arange(self.action_size+1)]]==0] = -torch.inf


        return logits_a, logits_q, loss
