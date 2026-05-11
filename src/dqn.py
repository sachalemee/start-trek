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
    def __init__(self, capacity: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.idx = 0

    def __len__(self):
        return len(self.states)

    def store(self, state, action, next_state, reward, done):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            self.states[self.idx] = state
            self.actions[self.idx] = action
            self.next_states[self.idx] = next_state
            self.rewards[self.idx] = reward
            self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batchsize):
        idx = random.sample(range(len(self.states)), k=batchsize)
        states = torch.from_numpy(np.array(self.states)[idx]).float().to(self.device)
        actions = torch.from_numpy(np.array(self.actions)[idx]).long().to(self.device)
        next_states = torch.from_numpy(np.array(self.next_states)[idx]).float().to(self.device)
        rewards = torch.from_numpy(np.array(self.rewards)[idx]).float().to(self.device)
        dones = torch.from_numpy(np.array(self.dones)[idx]).float().to(self.device)
        return states, actions, next_states, rewards, dones


class DuelingNetwork(nn.Module):

    def __init__(self, state_shape: int, action_shape: int, lr: float):
        super(DuelingNetwork, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(state_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


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
        learn_every: int = 4,
    ):
        self.discount = discount
        self.batchsize = batchsize
        self.grad_clip = grad_clip
        self.learn_every = learn_every
        self.step_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayMemory(memory_capacity)
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

    def learn(self, state, action, next_state, reward, done) -> None:
        self.memory.store(state, action, next_state, reward, done)
        self.step_count += 1
        if len(self.memory) < self.batchsize or self.step_count % self.learn_every != 0:
            return
        if self.last_update >= self.target_update:
            self.update_target()
        else:
            self.last_update += 1
        states, actions, next_states, rewards, dones = self.memory.sample(self.batchsize)
        q_pred = self.online(states).gather(1, actions.view(-1, 1))
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target(next_states).gather(1, next_actions)
            q_next = q_next.squeeze(1) * (1.0 - dones)
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
        self.online.eval()