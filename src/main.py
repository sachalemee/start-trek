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

from src.video import trigger_videos
from src.mode import Mode
from src.agent import Agent
from src.qlearn import QLearningAgent
from src.dqlearn import DQLearningAgent
from src.train import TrainMode
from src.eval import EvalMode
from src.parse import ParserConfig, RenderMode, RunMode


CSVPATH = "/csv"
EPPATH = CSVPATH + "/episodes"

def load_agent(cfg: ParserConfig, env) -> Agent:
    agent = Agent()
    if (cfg.algo == "qlearn"):
        agent = QLearningAgent(cfg.hyper.lr, cfg.hyper.discount, env.observation_space.shape)
    elif (cfg.algo == "dqn"):
        agent = DQLearningAgent()
    return agent

def build_mode(cfg: ParserConfig) -> Mode:
    if cfg.env.run_mode == RunMode.TRAIN:
        return TrainMode(cfg)
    return EvalMode(cfg)

def render(cfg: ParserConfig, args):
    render_mode = "rgb_array"
    if cfg.env.render_mode == RenderMode.HUMAN:
        render_mode = "human"
    
    df = pd.DataFrame(columns=["episode", "total_reward"])

    env = gym.make(
        "LunarLander-v3",
        max_episode_steps=cfg.env.max_episode_steps,
        render_mode=render_mode,
        enable_wind=cfg.env.enable_wind,
        wind_power=cfg.env.wind_power,
        turbulence_power=cfg.env.turbulence_power,
    )
 
    mode  = build_mode(cfg)
    agent = mode.load_agent(env)
 
    obs, _ = env.reset(seed=cfg.seed)
 
    if cfg.env.render_mode == RenderMode.RECORD:
        if not args.override and os.path.isdir("data_" + cfg.env.model):
            print("Error: could not override " + cfg.env.model + " folder")
            print("Use --override-data to override current data folder")
            exit(84)
        if args.override:
            shutil.rmtree("data_" + cfg.env.model)
        env = trigger_videos(env, cfg.env.trigger_video, cfg.env.model)
    try:
        epsilon = cfg.hyper.epsilon_start
        for episode in tqdm(range(cfg.env.nb_loop)):
            terminated = False
            truncated  = False
            r = 0
            ep_r = 0
            while not (terminated or truncated):
                env.render()
 
                prev_obs = obs
                action   = agent.act(obs, env.action_space.sample, mode.epsilon)
 
                obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                mode.step(agent, prev_obs, action, obs, r, done)

                ep_r += r

            df.loc[len(df)] = [episode, ep_r]
            obs, _ = env.reset()
        mode.end(agent)
    except KeyboardInterrupt:
        mode.end(agent)
    env.close()
    os.mkdir("data_" + cfg.env.model + CSVPATH)
    df.to_csv("data_" + cfg.env.model + EPPATH + ".csv", index=False)


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
