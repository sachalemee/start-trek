##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## main
##

import gymnasium as gym
import argparse
from tqdm import tqdm

from src.video import trigger_videos
from src.mode import Mode
from src.train import TrainMode
from src.eval import EvalMode
from src.parse import ParserConfig, RenderMode, RunMode

def build_mode(cfg: ParserConfig) -> Mode:
    if cfg.env.run_mode == RunMode.TRAIN:
        return TrainMode(cfg)
    return EvalMode(cfg)

def render(cfg: ParserConfig):
    render_mode = "rgb_array"
    if cfg.env.render_mode == RenderMode.HUMAN:
        render_mode = "human"
 
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
        env = trigger_videos(env, cfg.env.trigger_video)
 
    try:
        total_r = 0

        for _ in tqdm(range(cfg.env.nb_loop)):
            terminated = False
            truncated  = False
 
            while not (terminated or truncated):
                env.render()
 
                prev_obs = obs
                action   = agent.act(obs, env.action_space.sample, mode.epsilon)
 
                obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                mode.step(agent, prev_obs, action, obs, r, done)

                total_r += r

            success = mode.record_episode_score(total_r)

            obs, _ = env.reset()

            if success:
                break

        mode.end(agent)
    except KeyboardInterrupt:
        mode.end(agent)
    env.close()

def main(argv: list[str]):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument('file',
                        help="YAML configs (env, algo, hypers, seeds).",
                        type=str)
    args = parser.parse_args()

    render(ParserConfig(args.file))
