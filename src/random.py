##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## qlearn
##

import numpy as np
from src.agent import Agent

class RandomAgent(Agent):

    def policy(self, obs: list[8: float], fallback: callable, epsilon: float = 0.0) -> list[2: float]:
        return fallback()

    def learn(self, prev_obs, action, obs, r, terminated, truncated) -> None:
        pass