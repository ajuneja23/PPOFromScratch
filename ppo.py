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
    def __init__(self, env, lr):
        """
        state_dim=2 (xCoord,yCoord)
        action_dim=2(xMovement,yMovement)
        reward_dim=1
        """
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim)
        self.critic = Critic(self.obs_dim, 1)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)


    def work(self, num_steps,iterations_per_step):
        
        for step in range(num_steps):
            rewards_tot, rtg_tot, log_probs_tot, actions_tot, observations_tot = [], [], [], [], [], []


            rewards, rewards_to_go, log_probs, actions, observations = collectTrajectory(self.actor)
            rewards_tot.extend(rewards)
            rtg_tot.extend(rewards_to_go)
            log_probs_tot.extend(log_probs)
            actions_tot.extend(actions)
            observations_tot.extend(observations)
            #critic now needs to evaluate the 


            normalized_advantages,valueEstimates= self.getAdvantageEstimates(observations_tot, rtg_tot)

            for _ in range(iterations_per_step):
                
                rew, rewtg, curr_probs, a, obs= collectTrajectory(self.actor)
                clip = 0.2
                past_probs = log_probs_tot

                ratio = np.exp(curr_probs - past_probs)

                clip_term = torch.clamp(ratio, 1-clip, 1+clip)

                term1 = ratio * normalized_advantages

                term2 = ratio * clip_term

                actor_loss = (-1 * torch.min(term1, term2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                
                critic_loss=nn.MSELoss()
                res=critic_loss(valueEstimates,rtg_tot)
                self.critic_optim.zero_grad()
                res.backward()
                self.critic_optim.step()

            




        


    def getAdvantageEstimates(self, observations, rewards_to_go):
        criticInput = torch.tensor(observations)
        rewards_to_go = torch.tensor(rewards_to_go)
        valueScores=self.critic(criticInput).squeeze()
        advantageEstimates=rewards_to_go-valueScores.detach()

        normalizedAdvEstimates=(advantageEstimates-advantageEstimates.mean())/(advantageEstimates.std()+1e-10)
        return normalizedAdvEstimates,valueScores
    

    


        

        