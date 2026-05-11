##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## main
##

import gymnasium as gym
import argparse
import shutil
from tqdm import tqdm
import pandas as pd
import os

from src.video import trigger_videos
from src.mode import Mode
from src.train import TrainMode
from src.eval import EvalMode
from src.parse import ParserConfig, RenderMode, RunMode
from src.termination import classify_termination, lander_position

CSVPATH = "/csv"
EPPATH = CSVPATH + "/episodes"

def build_mode(cfg: ParserConfig) -> Mode:
    if cfg.env.run_mode == RunMode.TRAIN:
        return TrainMode(cfg)
    return EvalMode(cfg)

def render(cfg: ParserConfig, args):
    render_mode = "rgb_array"
    if cfg.env.render_mode == RenderMode.HUMAN:
        render_mode = "human"
    
    df = pd.DataFrame(columns=["episode", "total_reward", "cause", "x", "y"])

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
        if cfg.env.load and os.path.isdir("data_" + cfg.env.model):
            try:
                shutil.rmtree("data_" + cfg.env.model + CSVPATH)
                shutil.rmtree("data_" + cfg.env.model + "/videos")
            except FileNotFoundError:
                pass
        if not (args.override or cfg.env.load) and os.path.isdir("data_" + cfg.env.model):
            print("Error: could not override " + cfg.env.model + " folder")
            print("Use --override-data to override current data folder")
            exit(84)
        if args.override:
            try:
                shutil.rmtree("data_" + cfg.env.model)
            except FileNotFoundError:
                pass
        env = trigger_videos(env, cfg.env.trigger_video, cfg.env.model)
    try:
        nb_r = 0
        for episode in tqdm(range(mode.nb_loop)):
            ep_r = 0
            terminated = False
            truncated  = False
            while not (terminated or truncated):
                env.render()
 
                prev_obs = obs
                action   = agent.act(obs, env.action_space.sample, mode.epsilon)
 
                obs, r, terminated, truncated, _ = env.step(action)

                mode.step(agent, prev_obs, action, obs, r, terminated, truncated)

                ep_r += r

            cause = classify_termination(terminated, truncated, obs, r)
            x, y  = lander_position(obs)

            success = mode.record_episode_score(ep_r)

            df.loc[len(df)] = [episode, ep_r, cause.value, x, y]
            nb_r += 1
            obs, _ = env.reset()

            if (episode % cfg.env.mean_steps == 0
                and cfg.env.run_mode == RunMode.TRAIN
                and episode != 0):
                print(f"Mean reward over last {cfg.env.mean_steps}"
                      f" episodes from the {episode} episode: "
                      f"{sum(mode.episode_scores) / len(mode.episode_scores)}")

            if success:
                break

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