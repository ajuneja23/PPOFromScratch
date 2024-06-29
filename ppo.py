import actor.actor as Actor
import critic.critic as Critic
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from env_setup_and_trajectory_collection import collectTrajectory


class PPO:
    def __init__(self, env, lr,action_dim,state_dim):
        """
        state_dim=2 (xCoord,yCoord)
        action_dim=2(xMovement,yMovement)
        reward_dim=1
        """
        self.env = env
        #self.action_dim = env.action_space.shape[0]
        #self.obs_dim = env.observation_space.shape[0]
        self.action_dim=action_dim
        self.state_dim=state_dim
        self.actor = Actor(self.state_dim+1, self.action_dim)#position and most recent reward comprise the state_dim 
        self.critic = Critic(action_dim+state_dim, 1)#obs_dim=concatenated action-state of a move (I went to position 4,4 after taking action moveLeft)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.variance = torch.full(size=(self.action_dim,), fill_value=0.4)
        self.variance_matrix = torch.diag(self.variance)

    def work(num_iter):
        curr = 0

    def getAdvantageEstimates(self,rewards,rewards_to_go,states):
        criticInput=torch.cat((torch.tensor(states),torch.tensor(rewards)),1)
        valueScores=critic(criticInput).squeeze()
        advantageEstimates=rewards_to_go-valueScores.detach()

        normalizedAdvEstimates=(advantageEstimates-advantageEstimates.mean())/(advantageEstimates.std()+1e-10)
        return normalizedAdvEstimates


        

        