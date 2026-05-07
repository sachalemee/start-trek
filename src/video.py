##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-16
## File description:
## video
##

import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import glob

VIDEOPATH = "./videos"

def remove_videos():
    files = glob.glob(VIDEOPATH + "/*.mp4")
    for f in files:
        os.remove(f)
    try:
        os.removedirs(VIDEOPATH)
    except FileNotFoundError:
        pass

def trigger_videos(env: gym.Env, trigger: int) -> gym.Env:
    remove_videos()
    return RecordVideo(env, VIDEOPATH, episode_trigger=lambda t: ((t % trigger) == 0))