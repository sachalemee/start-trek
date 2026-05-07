##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## train
##

import gymnasium as gym

from src.mode import Mode
from src.parse import ParserConfig
from src.agent import Agent
from src.dqlearn import DQLearningAgent
from src.qlearn import QLearningAgent


class TrainMode(Mode):

    def __init__(self, cfg: ParserConfig):
        self.cfg = cfg
        self.epsilon = cfg.hyper.epsilon_start

    def load_agent(self, env: gym.Env) -> Agent:
        if self.cfg.algo == "qlearn":
            return QLearningAgent(
                self.cfg.hyper.lr,
                self.cfg.hyper.discount,
                env.observation_space.shape,
            )
        return DQLearningAgent(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space,
            memory_capacity=self.cfg.hyper.memory_capacity,
            discount=self.cfg.hyper.discount,
            lr=self.cfg.hyper.lr,
            batchsize=self.cfg.hyper.batchsize,
        )

    def step(self, agent: Agent, prev_obs, action, obs, reward, done) -> None:
        agent.learn(prev_obs, action, obs, reward, done)
        self.epsilon = max(
            self.cfg.hyper.epsilon_end,
            self.epsilon * self.cfg.hyper.epsilon_decay,
        )

    def end(self, agent: Agent) -> None:
        agent.save(self.cfg.env.model)