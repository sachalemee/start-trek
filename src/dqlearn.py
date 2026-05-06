##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-16
## File description:
## dqlearn
##

import random
from src.agent import Agent

class DQLearningAgent(Agent):
    def act(self, obs: list[8: float], fallback: callable) -> int:
        return random.randrange(4)

    def learn(self, obs: list[8: float], r: float | None) -> None:
        return