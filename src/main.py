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

NONE = 0
RECORD = 1
HUMAN = 2

display_mode = 0
render_mode = "rgb_array"
if display_mode == HUMAN:
    render_mode = "human"

env = gym.make("LunarLander-v3", render_mode=render_mode, continuous=True, enable_wind=True, wind_power=15.0, turbulence_power=1.5)

# Reset the environment to generate the first observation
obs, info = env.reset(seed = SEED)

trigger = lambda t: ((t % 10) == 0)

if display_mode == RECORD:
    env = RecordVideo(env, "./videos", episode_trigger=trigger)
def main():
    try:
        while True:
            env.render()
            action = env.action_space.sample()
            # this is where you would insert your policy
            # action = env.action_space.sample()

            # step (transition) through the environment with the action
            # receiving the next observation, reward and if the episode has terminated or truncated
            # obs, r, terminated, truncated, info = env.step(action)
            obs, r, terminated, truncated, info = env.step(action)

            # If the episode has ended then we can reset to start a new episode
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    env.close()
