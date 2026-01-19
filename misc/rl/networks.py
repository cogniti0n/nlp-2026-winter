import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], activation=nn.Tanh, output_activation=nn.Identity)

    def dist(self, obs_t: torch.Tensor) -> Categorical:
        logits = self.net(obs_t)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs_t: torch.Tensor):
        d = self.dist(obs_t)
        a = d.sample()
        logp = d.log_prob(a)
        return a.item(), logp.item()

class ValueFn(nn.Module):
    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1], activation=nn.Tanh, output_activation=nn.Identity)

    def forward(self, obs_t):
        return self.net(obs_t).squeeze(-1)

def discounted_returns(rews, gamma: float):
    out = []
    g = 0.0
    for r in reversed(rews):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return np.array(out, dtype=np.float32)

def rollout_episode(env, policy: CategoricalPolicy, device="cpu"):
    obs, _ = env.reset()
    obs_list, act_list, logp_list, rew_list = [], [], [], []
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        a, logp = policy.act(obs_t)
        nxt, r, term, trunc, _ = env.step(a)
        done = term or trunc
        obs_list.append(obs)
        act_list.append(a)
        logp_list.append(logp)
        rew_list.append(r)
        obs = nxt
    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.int64),
        np.array(logp_list, dtype=np.float32),
        np.array(rew_list, dtype=np.float32),
    )
