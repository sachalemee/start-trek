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


class Network(nn.Module):
    def __init__(self, state_shape, action_shape, lr):
        super(Network, self).__init__()
        self.dense1 = nn.Linear(in_features=state_shape, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=64)
        self.dense3 = nn.Linear(in_features=64, out_features=action_shape)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class DQN(Agent):

    def __init__(self, state_shape, action_shape, memory_capacity, discount, lr, batchsize, target_update):
        self.discount = discount
        self.batchsize = batchsize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(memory_capacity)
        self.online = Network(state_shape[0], action_shape.n, lr).to(self.device)
        self.target = Network(state_shape[0], action_shape.n, lr).to(self.device)
        self.target.eval()
        self.update_target()
        self.target_update = target_update
        self.last_update = 0

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())
        self.last_update = 0

    def act(self, obs, fallback: callable, epsilon: float = 0.0):
        if random.random() < epsilon:
            return fallback()
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action = torch.argmax(self.online(obs))
        return action.item()

    def learn(self, state, action, next_state, reward, done) -> None:
        self.memory.store(state, action, next_state, reward, done)
        if len(self.memory) < self.batchsize:
            return
        if self.target_update < self.last_update:
            self.update_target()
        else:
            self.last_update += 1

        states, actions, next_states, rewards, dones = self.memory.sample(self.batchsize)

        q_pred = self.online.forward(states).gather(1, actions.view(-1, 1))

        q_target = self.target.forward(next_states).max(dim=1).values
        q_target = q_target * (1.0 - dones)

        y_j = (rewards + self.discount * q_target).view(-1, 1)

        self.online.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred)
        loss.backward()
        self.online.optimizer.step()

    def save(self, filename) -> None:
        torch.save(self.online.state_dict(), filename)
    
    def load(self, filename) -> None:
        self.online.load_state_dict(torch.load(filename))
        self.online.eval()