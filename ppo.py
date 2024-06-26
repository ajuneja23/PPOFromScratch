import actor.actor as Actor
import critic.critic as Critic
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal


class PPO:
    def __init__(self, env, lr):

        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]

        self.actor = Actor(self.action_dim, self.obs_dim)
        self.critic = Critic(self.obs_dim, 1)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.variance = torch.full(size=(self.action_dim,), fill_value=0.4)
        self.variance_matrix = torch.diag(self.variance)

    def work(num_iter):
        curr = 0
        

        