##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## qlearn
##

import numpy as np
from src.agent import Agent
from math import ceil

EZOOM_VAL = 1
AZOOM_VAL = 10

states_d = (5,5,20,20,15,20,2,2)
action_d = (20,20)

# lr = 0.8 # a quel point tu favorise les nouvelles infos
# discount_factor = 0.95 # valuation des récompenses
# exploration_prob = 0.2 # exploration

# ED = [2.5, 2.5, 10., 10., 6.2831855, 10.]
# n_states = prod([round(e * EZOOM_VAL) * 2 for e in ED])

# AD = [1., 1.]
# n_actions = prod([round(e * AZOOM_VAL) * 2 for e in AD])

class QLearningAgent(Agent):

    Qtable: np.array = np.zeros(states_d + action_d)

    lr: float
    discount: float
    exploration: float

    ls: list[6: int] = []
    la: list[2: int] = []

    def __init__(self, lr, discount, exploration):
        self.lr = lr
        self.discount = discount
        self.exploration = exploration
    
    def process(self, obs: list[8: float]) -> list[6: int]:
        return tuple([round(e * EZOOM_VAL) for e in obs])

    def act(self, obs: list[8: float], fallback: callable) -> list[2: float]:
        self.ls = self.process(obs)
        action = fallback()
        if np.random.rand() > self.exploration and np.max(self.Qtable[self.ls]) > 0.:
            action = divmod(np.argmax(self.Qtable[self.ls]), 20)
            self.la = tuple(action)
        else:
            action = fallback()
            self.la = tuple([round(e * AZOOM_VAL) for e in action])
        return action

    def learn(self, obs: list[8: float], r: float | None) -> None:
        """
        Calculate the best move to get the highest reward

        Parameters:
          obs: the observation state as an 8 dimentional vector
          r: the reward of the last episode, or None if it is the first episode
        """
        obs = self.process(obs)
        future_optimal_value = np.max(self.Qtable[obs])
        learned_value = r + self.discount * future_optimal_value

        old_value = self.Qtable[self.ls][self.la]
        self.Qtable[self.ls][self.la] = (1-self.lr) * old_value + self.lr * learned_value

