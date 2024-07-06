import gymnasium as gym

import PPO from PPO


env=gym.make('LunarLanderContinuous-v2',render_mode='human')


lr=0.005
model=PPO(env,lr)

model.work(10,5)

