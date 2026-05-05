##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## main
##

import src.qlearn as qlearn

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import sys
import os
import glob

SEED = 67
VIDEOPATH = "./videos"
DEFAULT_NBLOOP = 100
DEFAULT_MAX_STEP = 100

NONE = 0
RECORD = 1
HUMAN = 2

def remove_videos():
    files = glob.glob(VIDEOPATH + "/*.mp4")
    for f in files:
        os.remove(f)
    try:
        os.removedirs(VIDEOPATH)
    except FileNotFoundError:
        pass

def help(progname: str):
    print("usage: ", progname, " [--mode {none/record/human}] [--nb-loop {int}]")
    sys.exit(1)

def parse_argv(argv: list[str]) -> list[3: int]:
    display_mode = RECORD
    nb_loop = DEFAULT_NBLOOP
    max_step = DEFAULT_MAX_STEP
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
        pass

    try:
        nb_loop = int(argv[argv.index("--nb-loop") + 1])
    except IndexError:
        help(argv[0])
    except ValueError:
        pass
    if nb_loop > 1:
        nb_loop += 1

    try:
        max_step = int(argv[argv.index("--max-episode-step") + 1])
    except IndexError:
        help(argv[0])
    except ValueError:
        pass
    
    return display_mode, nb_loop, max_step

def main(argv: list[str]):
    display_mode, nb_loop, max_step = parse_argv(argv)

    render_mode = "rgb_array"
    if display_mode == HUMAN:
        render_mode = "human"
    
    env = gym.make("LunarLander-v3", max_episode_steps=max_step, render_mode=render_mode, continuous=True, enable_wind=True, wind_power=15.0, turbulence_power=1.5)

    obs, _ = env.reset(seed = SEED)

    agent: qlearn.Agent = qlearn.QLearningAgent(0.8, 0.95, 0.2)
    
    if display_mode == RECORD:
        remove_videos()
        env = RecordVideo(env, VIDEOPATH, episode_trigger=lambda t: ((t % 10) == 0))
    try:
        for _ in range(nb_loop):
            terminated = 0
            truncated = 0
            r = 0
            while not (terminated or truncated):
                env.render()

                action = agent.act(obs, env.action_space.sample)

                obs, r, terminated, truncated, _ = env.step(action)

                agent.learn(obs, r)

            obs, _ = env.reset()

    except KeyboardInterrupt:
        pass
    env.close()
