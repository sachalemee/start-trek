##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## main
##

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import sys
import os
import glob

SEED = 67
VIDEOPATH = "./videos"
DEFAULT_NBLOOP = 100

NONE = 0
RECORD = 1
HUMAN = 2

def help(progname: str):
    print("usage: ", progname, " [--mode {none/record/human}] [--nb-loop {int}]")
    sys.exit(1)

def main(argv: list[str]):
    display_mode = NONE
    nb_loop = 0

    try:
        mode = argv[argv.index("--mode") + 1]
        if mode.lower() == "none":
            display_mode = NONE
        elif mode.lower() == "record":
            display_mode = RECORD
        elif mode.lower() == "human":
            display_mode = HUMAN
        else:
            help(argv[0])
    except IndexError:
        help(argv[0])
    except ValueError:
        display_mode = RECORD

    try:
        nb_loop = int(argv[argv.index("--nb-loop") + 1])
    except IndexError:
        help(argv[0])
    except ValueError:
        nb_loop = DEFAULT_NBLOOP
    
    render_mode = "rgb_array"
    if display_mode == HUMAN:
        render_mode = "human"
    
    env = gym.make("LunarLander-v3", render_mode=render_mode, continuous=True, enable_wind=True, wind_power=15.0, turbulence_power=1.5)

    env.reset(seed = SEED)
    
    if display_mode == RECORD:
        files = glob.glob(VIDEOPATH + "/*.mp4")
        for f in files:
            os.remove(f)
        os.removedirs(VIDEOPATH)
        env = RecordVideo(env, VIDEOPATH, episode_trigger=lambda t: ((t % 10) == 0))
    try:
        for _ in range(nb_loop + 1):
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
                    break
            env.reset()
    except KeyboardInterrupt:
        pass
    env.close()
