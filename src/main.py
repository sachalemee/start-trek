##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## main
##

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import sys
import argparse
import os
import glob
from tqdm import tqdm

from src.qlearn import QLearningAgent
from src.dqlearn import DQLearningAgent
from src.agent import Agent

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

def load_agent(cfg: ParserConfig) -> Agent:
    agent = Agent()
    if (cfg.algo == "qlearn"):
        agent = QLearningAgent(0.8, 0.95)
    elif (cfg.algo == "dqn"):
        agent = DQLearningAgent()
    return agent

def render(cfg: ParserConfig):
    render_mode = "rgb_array"
    if cfg.env.render_mode == RenderMode.HUMAN:
        render_mode = "human"
    
    agent = load_agent(cfg)

    env = gym.make("LunarLander-v3",
                   max_episode_steps=cfg.hyper.max_episode_steps,
                   render_mode=render_mode,
                   enable_wind=cfg.hyper.enable_wind,
                   wind_power=cfg.hyper.wind_power,
                   turbulence_power=cfg.hyper.turbulence_power
                )

    obs, _ = env.reset(seed = cfg.seed)
    
    if cfg.env.render_mode == RenderMode.RECORD:
        remove_videos()
        env = RecordVideo(env, VIDEOPATH, episode_trigger=lambda t: ((t % 10) == 0))
    try:
        total_r = 0
        nb_r = 0
        epsilon = 1.0
        for _ in tqdm(range(cfg.env.nb_loop)):
            terminated = 0
            truncated = 0
            r = 0
            while not (terminated or truncated):
                env.render()

                action = agent.act(obs, env.action_space.sample, epsilon)

                obs, r, terminated, truncated, _ = env.step(action)

                agent.learn(obs, r)

                epsilon = max(0.0, epsilon * 0.995)

                total_r += r

            nb_r += 1
            obs, _ = env.reset()

    except KeyboardInterrupt:
        pass
    env.close()
    print(total_r / nb_r)

def main(argv: list[str]):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument('file',
                        help="YAML configs (env, algo, hypers, seeds).",
                        type=str)
    args = parser.parse_args()

    render(ParserConfig(args.file))
