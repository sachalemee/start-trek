##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## main
##

import gymnasium as gym
import sys

SEED = 67

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human", continuous=True, enable_wind=True, wind_power=15.0, turbulence_power=1.5)

# Reset the environment to generate the first observation
obs, info = env.reset(seed = SEED)

try:
    while True:
        # this is where you would insert your policy
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        obs, r, terminated, truncated, info = env.step(action)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    sys.exit(0)

env.close()
