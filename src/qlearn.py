##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## qlearn
##

import numpy as np
from src.agent import Agent

EZOOM_VAL = 1

states_d = (5,5,20,20,15,20,2,2)

# lr = 0.8 # a quel point tu favorise les nouvelles infos
# discount_factor = 0.95 # valuation des récompenses
# exploration_prob = 0.2 # exploration

# ED = [2.5, 2.5, 10., 10., 6.2831855, 10.]
# n_states = prod([round(e * EZOOM_VAL) * 2 for e in ED])

# AD = [1., 1.]
# n_actions = prod([round(e * AZOOM_VAL) * 2 for e in AD])

class QLearningAgent(Agent):


    Qtable: np.array
    lr: float
    discount: float

    ls: list[6: int] = []
    la: int = -1

    def __init__(self, lr, discount, action_shape):
        self.lr = lr
        self.discount = discount
        self.action_shape = action_shape
        self.Qtable = np.zeros(states_d + action_shape);
    
    def process(self, obs: list[8: float]) -> list[6: int]:
        return tuple([round(e * EZOOM_VAL) for e in obs])

    def act(self, obs: list[8: float], fallback: callable, epsilon: float) -> list[2: float]:
        self.ls = self.process(obs)
        if np.random.rand() > epsilon and np.max(self.Qtable[self.ls]) > 0.:
            action = np.argmax(self.Qtable[self.ls])
            self.la = action
        else:
            action = fallback()
            self.la = action
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

