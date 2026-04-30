##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## main
##

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import sys

SEED = 67

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="rgb_array", continuous=True, enable_wind=True, wind_power=15.0, turbulence_power=1.5)

# Reset the environment to generate the first observation
obs, info = env.reset(seed = SEED)

trigger = lambda t: ((t % 10) == 0)
recording_env = RecordVideo(env, "./videos", episode_trigger=trigger)

observation = recording_env.reset()

try:
    while True:
        recording_env.render()
        action = recording_env.action_space.sample()
        # this is where you would insert your policy
        # action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        # obs, r, terminated, truncated, info = env.step(action)
        obs, r, terminated, truncated, info = recording_env.step(action)


        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    recording_env.close()
    sys.exit(0)

env.close()
