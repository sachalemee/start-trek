##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-16
## File description:
## video
##

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

VIDEOPATH = "/videos"

def trigger_videos(env: gym.Env, trigger: int, path: str) -> gym.Env:
    return RecordVideo(env, "data_" + path + VIDEOPATH, episode_trigger=lambda t: ((t % trigger) == 0))