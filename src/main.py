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
import argparse
import os
import glob
from tqdm import tqdm

from src.parse import ParserConfig, HyperConfig, EnvConfig, RenderMode

VIDEOPATH = "./videos"

def remove_videos():
    files = glob.glob(VIDEOPATH + "/*.mp4")
    for f in files:
        os.remove(f)
    try:
        os.removedirs(VIDEOPATH)
    except FileNotFoundError:
        pass

def render(cfg: ParserConfig):
    render_mode = "rgb_array"
    if cfg.env.render_mode == RenderMode.HUMAN:
        render_mode = "human"
    
    env = gym.make("LunarLander-v3",
                   max_episode_steps=cfg.hyper.max_episode_steps,
                   render_mode=render_mode,
                   continuous=True,
                   enable_wind=cfg.hyper.enable_wind,
                   wind_power=cfg.hyper.wind_power,
                   turbulence_power=cfg.hyper.turbulence_power
                )

    obs, _ = env.reset(seed = cfg.seed)

    agent: qlearn.Agent = qlearn.QLearningAgent(0.8, 0.95, 0.2)
    
    if cfg.env.render_mode == RenderMode.RECORD:
        remove_videos()
        env = RecordVideo(env, VIDEOPATH, episode_trigger=lambda t: ((t % 10) == 0))
    try:
        for _ in tqdm(range(cfg.env.nb_loop)):
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

def main(argv: list[str]):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument('file',
                        help="YAML configs (env, algo, hypers, seeds).",
                        type=str)
    args = parser.parse_args()

    try:
        cfg = ParserConfig()
        cfg.load(args.file)
        render(cfg)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        print(f"StartTrek error: {e}", file=sys.stderr)
        sys.exit(84)
