##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## qlearn
##

import numpy as np
from src.agent import Agent
from src.discretizer import Discretizer

class QLearningAgent(Agent):

    Qtable: np.array
    lr: float
    discount: float
    discretizer: Discretizer

    ls: list[6: int] = []

    def __init__(self, lr, discount, discretizer: Discretizer = None):
        self.lr = lr
        self.discount = discount
        
        # use provide discretizer/create default one
        if discretizer is None:
            self.discretizer = Discretizer(states_d, EZOOM_VAL)
        else:
            self.discretizer = discretizer
            
        # initialize qtable with discretizer's state shape
        self.Qtable = np.zeros(self.discretizer.get_state_shape() + (4,))
    
    def process(self, obs: list[8: float]) -> tuple:
        return self.discretizer.discretize(obs)

    def policy(self, obs: list[8: float], fallback: callable, epsilon: float = 0.0) -> list[2: float]:
        self.ls = self.process(obs)
        if np.random.rand() > epsilon:
            action = np.argmax(self.Qtable[self.ls])
        else:
            action = fallback()
        return action

    def learn(self, prev_obs, action, obs, r, terminated, truncated) -> None:
        obs = self.process(obs)

        future_optimal_action = np.max(self.Qtable[obs])
        if terminated or truncated:
            self.Qtable[obs] += self.lr * (r + future_optimal_action - self.Qtable[obs])
        else:
            self.Qtable[obs] += self.lr * (r - self.Qtable[obs])

    def save(self, filename) -> None:
        np.save(filename, self.Qtable)
    
    def load(self, filename) -> None:
        self.Qtable = np.load(filename)