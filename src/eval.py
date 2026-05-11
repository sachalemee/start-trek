##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## eval
##

import gymnasium as gym
from collections import deque

from src.mode import Mode
from src.parse import ParserConfig
from src.agent import Agent
from src.dqn import DQNAgent
from src.qlearn import QLearningAgent


class EvalMode(Mode):

    def __init__(self, cfg: ParserConfig):
        self.cfg = cfg
        self.epsilon = 0.0
        self.episode_scores = deque(maxlen=cfg.env.mean_steps)
        self.nb_loop = cfg.env.mean_steps

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
        agent.load(self.cfg.env.model)
        return agent

    def step(self, agent: Agent, prev_obs, action, obs, reward, terminated, truncated) -> None:
        pass

    def record_episode_score(self, episode_score: float) -> bool:
        self.episode_scores.append(episode_score)

    def end(self, agent: Agent) -> None:
        print(f"Mean score for {len(self.episode_scores)} episodes: {sum(self.episode_scores) / len(self.episode_scores)}")