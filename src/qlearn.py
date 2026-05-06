##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## qlearn
##

import numpy as np
from src.agent import Agent
from math import prod

ZOOM_VAL = 3

# Qtable += [(np.zeros(shape=8) * 10).astype(int), [0., 0.], 0.]

# learning_rate = 0.8 # a quel point tu favorise les nouvelles infos
# discount_factor = 0.95 # valuation des récompenses
# exploration_prob = 0.2 # exploration
D = [ 2.5, 2.5, 10., 10., 6.2831855, 10.]
n_states = prod([round(e * ZOOM_VAL) * 2 for e in D])

class QLearningAgent(Agent):

    Qtable: np.array = np.zeros(n_states)

    learning_rate: float
    discount: float
    exploration: float

    last_obs: list[8: float] = []
    last_action: list[2: float] = []

    def __init__(self, learning_rate, discount, exploration):
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration = exploration
    
    def process(self, obs: list[8: float]) -> list[6: int]:
        obs = [round(e * ZOOM_VAL) for e in obs]
        return obs[:-2]

    def act(self, obs: list[8: float], fallback: callable) -> list[2: float]:
        self.last_obs = self.process(obs)
        action = fallback()
        self.last_action = action
        return action

    def learn(self, obs: list[8: float], r: float | None) -> None:
        """
        Calculate the best move to get the highest reward

        Parameters:
          obs: the observation state as an 8 dimentional vector
          r: the reward of the last episode, or None if it is the first episode
        """

        obs = self.process(obs)
        self.Qtable[self.last_obs] += self.learning_rate * \
            (r + self.discount * self.Qtable[obs] - self.Qtable[self.last_obs])