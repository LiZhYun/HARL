import math
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.utils.models_tools import init, get_clones, check
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase, MLPLayer
from harl.models.base.rnn import RNNLayer
from harl.models.base.distributions import DiagGaussian, FixedNormal
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Action_Attention(nn.Module):
    def __init__(self, args, num_agents, action_space, obs_space, device=torch.device("cpu")):
        super(Action_Attention, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                obs_shape, hidden_sizes[0], activation_func
            )
            feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]

        self.num_agents = num_agents
        self.discrete = False
        self.sigmoid_gain = 0.3
        self.std_x_coef = 1
        self.std_y_coef = 0.75
        if action_space.__class__.__name__ == "Discrete":
            self.discrete = True # 1 0.95
            self.action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0] # 1 0.75

        self.net = PlainMLP(
            [feature_dim + self.action_dim] + list(hidden_sizes), activation_func, final_activation_func
        )

        self.layers = nn.ModuleList()
        for layer in range(2):
            self.layers.append(MixerBlock(args, self.num_agents,
                        hidden_sizes[-1], 
                        token_factor=0.5,
                        channel_factor=4))
                
        self.layer_norm = nn.LayerNorm(hidden_sizes[-1])

        self.head = init_(nn.Linear(hidden_sizes[-1], self.action_dim))

        self.to(device)
        self.turn_off_grad()

    def forward(self, x, state):
        x = check(x).to(**self.tpdv)
        state = check(state).to(**self.tpdv)

        if self.feature_extractor is not None:
            obs = self.feature_extractor(state)
        else:
            obs = state

        x = self.net(torch.cat([obs, x], -1))

        for layer in range(2):
            x = self.layers[layer](x)
        x = self.layer_norm(x)

        bias_ = self.head(x)
        log_std = torch.clamp(bias_, LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(log_std)
        # log_std = bias_ * self.std_x_coef
        # action_std = 1 / (1 + torch.exp(-self.sigmoid_gain * (log_std / self.std_x_coef))) * self.std_y_coef

        if self.discrete:
            # bias_ = bias_ - bias_.logsumexp(dim=-1, keepdim=True)
            # action_std = None
            action_std = -torch.log(-torch.log(action_std))
        # else:
        #     action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef

        return bias_, action_std
    
    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.parameters():
            p.requires_grad = False
    

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, args, num_agents, dims, 
                 dropout=0, token_factor=0.5, channel_factor=4):
        super().__init__()
        self.dims = dims
        self.token_layernorm = nn.LayerNorm(dims)
        token_dim = int(token_factor*dims) if token_factor != 0 else 1
        self.token_forward = FeedForward(num_agents, token_dim, dropout)
            
        self.channel_layernorm = nn.LayerNorm(dims)
        channel_dim = int(channel_factor*dims) if channel_factor != 0 else 1
        self.channel_forward = FeedForward(self.dims, channel_dim, dropout)

        self.dropout_1 = nn.Dropout(0.)
        self.dropout_2 = nn.Dropout(0.)
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1) # (10,64,2)
        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        # if obs_rep != None:
        #     x = x + obs_rep
        x = x + self.dropout_1(self.token_mixer(x)) # (10,2,64)
        x = x + self.dropout_2(self.channel_mixer(x))
        return x

class HyperBlock(nn.Module):
    def __init__(self, num_agents, action_dim, dims, 
                 dropout=0):
        super().__init__()
        self.dims = dims
        self.hyper_w1 = nn.Sequential(nn.Linear(num_agents*dims, dims),
                                          nn.ReLU(),
                                          nn.Linear(dims, num_agents*1))
        self.hyper_w12 = nn.Sequential(nn.Linear(num_agents*dims, dims),
                                          nn.ReLU(),
                                          nn.Linear(dims, num_agents*1))
        self.hyper_w2 = nn.Sequential(nn.Linear(num_agents*dims, dims),
                                        nn.ReLU(),
                                        nn.Linear(dims, dims*4*dims))
        self.hyper_w22 = nn.Sequential(nn.Linear(num_agents*dims, dims),
                                        nn.ReLU(),
                                        nn.Linear(dims, dims*4*dims))
        self.hyper_b1 = nn.Linear(num_agents*dims, 1)
        self.hyper_b12 = nn.Linear(num_agents*dims, num_agents)
        self.hyper_b2 = nn.Linear(num_agents*dims, 4*dims)
        self.hyper_b22 = nn.Linear(num_agents*dims, dims)

    def forward(self, x, obs_rep):
        bs, n_agents, action_dim = x.shape
        w1 = self.hyper_w1(obs_rep.view(bs, -1)).view(bs, n_agents, -1) # (3,2,1)
        b1 = self.hyper_b1(obs_rep.view(bs, -1)).view(bs, 1, -1)  # (3,1,1)
        hidden = F.relu(torch.bmm(x.view(bs, -1, n_agents), w1) + b1)  # (3,64,1)

        w12 = self.hyper_w12(obs_rep.view(bs, -1)).view(bs, -1, n_agents) # (3,1,2)
        b12 = self.hyper_b12(obs_rep.view(bs, -1)).view(bs, 1, -1)  # (3,1,1)
        hidden = F.relu(torch.bmm(hidden, w12) + b12).view(bs, n_agents, -1)  # (3,2,64)

        w2 = self.hyper_w2(obs_rep.view(bs, -1)).view(bs, self.dims, 4*self.dims)  # (3,64,4*64)
        b2 = self.hyper_b2(obs_rep.view(bs, -1)).view(bs, 1, 4*self.dims)  # (3,1,4*64)
        hidden = F.relu(torch.bmm(hidden, w2) + b2)  # (3, 2, 4*64)

        w22 = self.hyper_w22(obs_rep.view(bs, -1)).view(bs, self.dims*4, self.dims)  # (3,4*64,64)
        b22 = self.hyper_b22(obs_rep.view(bs, -1)).view(bs, 1, self.dims)  # (3,1,64)
        x = x + F.relu(torch.bmm(hidden, w22) + b22)  # (3, 2, 64)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.0, use_orthogonal=True, activation_id=1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.linear_1 = nn.Sequential(
            init_(nn.Linear(d_model, d_ff)), 
            nn.ReLU(), 
            nn.LayerNorm(d_ff))
        self.dropout1 = nn.Dropout(dropout)
        self.linear_2 = init_(nn.Linear(d_ff, d_model))  

    def forward(self, x):
        x = self.dropout1(self.linear_1(x))
        x = self.linear_2(x)
        return x


def ScaledDotProductAttention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.0, use_orthogonal=True):
        super(MultiHeadAttention, self).__init__()
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = init_(nn.Linear(d_model, d_model))
        self.v_linear = init_(nn.Linear(d_model, d_model))
        self.k_linear = init_(nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.out = init_(nn.Linear(d_model, d_model))

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention
        scores = ScaledDotProductAttention(
            q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0, use_orthogonal=True, activation_id=False):
        super(EncoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.attn1 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.ff = FeedForward(d_model, d_model, dropout, use_orthogonal, activation_id)

    def forward(self, x, obs_rep, mask=None):
        x = self.norm_1(x + self.attn1(x, x, x))
        x = self.norm_2(obs_rep + self.attn2(k=x, v=x, q=obs_rep))
        x = self.norm_3(x + self.ff(x))

        return x
