##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-16
## File description:
## dqlearn
##

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from src.agent import Agent

class ReplayMemory:
    def __init__(self, capacity: int, state_shape: tuple):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.size = 0
        self.idx = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def store(self, state, action, next_state, reward, terminated) -> None:
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx]= next_state
        self.rewards[self.idx] = reward
        self.terminated[self.idx] = float(terminated)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batchsize: int):
        idx = np.random.randint(0, self.size, size=batchsize)
        states = torch.from_numpy(self.states[idx]).to(self.device)
        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        terminated = torch.from_numpy(self.terminated[idx]).to(self.device)
        return states, actions, next_states, rewards, terminated


class DuelingNetwork(nn.Module):
    def __init__(self, state_shape: int, action_shape: int, lr: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_shape, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_shape),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)


class DQNAgent(Agent):

    def __init__(
        self,
        state_shape,
        action_shape,
        memory_capacity: int,
        discount: float,
        lr: float,
        batchsize: int,
        target_update: int,
        grad_clip: float = 10.0,
        learn_every: int = 2,
    ):
        self.discount = discount
        self.batchsize = batchsize
        self.grad_clip = grad_clip
        self.learn_every = learn_every
        self.step_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayMemory(memory_capacity, state_shape)
        self.online = DuelingNetwork(state_shape[0], action_shape.n, lr).to(self.device)
        self.target = DuelingNetwork(state_shape[0], action_shape.n, lr).to(self.device)
        self.target.eval()
        self.update_target()

        self.target_update = target_update
        self.last_update = 0

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.last_update = 0

    def act(self, obs, fallback: callable, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return fallback()
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            action = self.online(obs).squeeze(0).argmax()
        return action.item()

    def learn(self, state, action, next_state, reward, terminated, truncated) -> None:
        self.memory.store(state, action, next_state, reward, terminated)
        self.step_count += 1

        if len(self.memory) < self.batchsize or self.step_count % self.learn_every != 0:
            return

        if self.last_update >= self.target_update:
            self.update_target()
        else:
            self.last_update += 1

        states, actions, next_states, rewards, terminated_batch = self.memory.sample(self.batchsize)

        q_pred = self.online(states).gather(1, actions.view(-1, 1))

        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target(next_states).gather(1, next_actions).squeeze(1)
            q_next = q_next * (1.0 - terminated_batch)
            y_j = (rewards + self.discount * q_next).view(-1, 1)

        loss = F.huber_loss(q_pred, y_j)
        self.online.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.online.optimizer.step()

    def save(self, filename: str) -> None:
        torch.save(self.online.state_dict(), filename)

    def load(self, filename: str) -> None:
        self.online.load_state_dict(torch.load(filename, map_location=self.device))
        self.update_target()
        self.online.train()