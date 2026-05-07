##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## main
##

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import argparse
import shutil
from tqdm import tqdm
import pandas as pd
import os

from src.qlearn import QLearningAgent
from src.dqlearn import DQLearningAgent
from src.agent import Agent

from src.parse import ParserConfig, RenderMode

DATAPATH = "data"
VIDEOPATH = DATAPATH + "/videos"
CSVPATH = DATAPATH + "/csv"
EPPATH = CSVPATH + "/episodes"

def load_agent(cfg: ParserConfig, env) -> Agent:
    agent = Agent()
    if (cfg.algo == "qlearn"):
        agent = QLearningAgent(cfg.hyper.lr, cfg.hyper.discount, env.observation_space.shape)
    elif (cfg.algo == "dqn"):
        agent = DQLearningAgent()
    return agent

def render(cfg: ParserConfig, args):
    render_mode = "rgb_array"
    if cfg.env.render_mode == RenderMode.HUMAN:
        render_mode = "human"
    
    df = pd.DataFrame(columns=["episode", "total_reward"])

    env = gym.make("LunarLander-v3",
        max_episode_steps=cfg.env.max_episode_steps,
        render_mode=render_mode,
        enable_wind=cfg.env.enable_wind,
        wind_power=cfg.env.wind_power,
        turbulence_power=cfg.env.turbulence_power
    )

    agent = load_agent(cfg, env)

    obs, _ = env.reset(seed = cfg.seed)
    
    if cfg.env.render_mode == RenderMode.RECORD:
        if not args.override and os.path.isdir(DATAPATH):
            print("Error: could not override " + DATAPATH + " folder")
            print("Use --override-data to override current data folder")
            exit(84)
        if args.override:
            shutil.rmtree(DATAPATH)
        env = RecordVideo(env, VIDEOPATH, episode_trigger=lambda t: ((t % 10) == 0))
    try:
        epsilon = cfg.hyper.epsilon_start
        for episode in tqdm(range(cfg.env.nb_loop)):
            terminated = 0
            truncated = 0
            r = 0
            ep_r = 0
            while not (terminated or truncated):
                env.render()

                action = agent.act(obs, env.action_space.sample, epsilon)

                obs, r, terminated, truncated, _ = env.step(action)

                agent.learn(obs, r)

                epsilon = max(cfg.hyper.epsilon_end, epsilon * cfg.hyper.epsilon_decay)

                ep_r += r

            df.loc[len(df)] = [episode, ep_r]
            obs, _ = env.reset()

    except KeyboardInterrupt:
        pass
    env.close()
    os.mkdir(CSVPATH)
    df.to_csv(EPPATH + ".csv", index=False)


def main(argv: list[str]):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument('file',
                        help="YAML configs (env, algo, hypers, seeds).",
                        type=str)
    parser.add_argument('--override-data',
                        dest='override',
                        action='store_true',
                        help="override the data folder")
    parser.set_defaults(override=False)
    args = parser.parse_args()

    render(ParserConfig(args.file), args)
