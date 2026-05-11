##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## train
##

import gymnasium as gym
from pathlib import Path
from collections import deque

from src.mode import Mode
from src.parse import ParserConfig
from src.agent import Agent
from src.dqn import DQNAgent
from src.qlearn import QLearningAgent

class TrainMode(Mode):

    def __init__(self, cfg: ParserConfig):
        self.cfg = cfg
        self.epsilon = cfg.hyper.epsilon_start
        self.episode_scores = deque(maxlen=cfg.env.mean_steps)
        self.nb_loop = cfg.env.nb_loop

    def load_agent(self, env: gym.Env) -> Agent:
        if self.cfg.algo == "qlearn":
            agent = QLearningAgent(
                self.cfg.hyper.lr,
                self.cfg.hyper.discount,
                env.observation_space.shape,
            )
        else:
            agent = DQNAgent(
                state_shape=env.observation_space.shape,
                action_shape=env.action_space,
                memory_capacity=self.cfg.hyper.memory_capacity,
                discount=self.cfg.hyper.discount,
                lr=self.cfg.hyper.lr,
                batchsize=self.cfg.hyper.batchsize,
                target_update=self.cfg.hyper.target_update,
            )
        file = Path(self.cfg.env.model)
        if self.cfg.env.load and file.is_file():
            agent.load(self.cfg.env.model)
        return agent

    def step(self, agent: Agent, prev_obs, action, obs, reward, terminated, truncated) -> None:
        agent.learn(prev_obs, action, obs, reward, terminated, truncated)
        self.epsilon = max(
            self.cfg.hyper.epsilon_end,
            self.epsilon * self.cfg.hyper.epsilon_decay,
        )

    def record_episode_score(self, episode_score: float) -> bool:
        self.episode_scores.append(episode_score)
        if len(self.episode_scores) >= self.cfg.env.mean_steps:
            mean_score = sum(self.episode_scores) / len(self.episode_scores)
            if mean_score >= self.cfg.env.mean:
                print(f"Mean score : {mean_score} >= {self.cfg.env.mean}")
                return True
        return False

    def end(self, agent: Agent) -> None:
        agent.save("data_" + self.cfg.env.model + "/model")