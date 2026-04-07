import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class Agent(nn.Module):
    def __init__(self, num_tables=3):
        super().__init__()

        self.num_tables = num_tables
        self.join_types = 3
        self.use_index = 2

        # total actions = 3 * 3 * 2 = 18
        self.action_size = num_tables * self.join_types * self.use_index

        # state size (approx)
        self.state_size = num_tables * 5 + 1  # adjust if needed

        self.net = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        # RL params
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.gamma = 0.95

    def forward(self, x):
        return self.net(x)

    # -------- Encode state --------
    def encode_state(self, obs):
        state = []

        state.extend(obs.table_rows)        # num_tables values
        state.extend(obs.selectivities)     # num_tables values
        state.extend(obs.has_index)         # num_tables values

        # chosen order (pad with -1 to fixed length)
        padded_chosen = obs.chosen_order + [-1] * (self.num_tables - len(obs.chosen_order))
        state.extend(padded_chosen)         # num_tables values

        # remaining tables (pad with -1 to fixed length)
        padded_remaining = obs.remaining_tables + [-1] * (self.num_tables - len(obs.remaining_tables))
        state.extend(padded_remaining)      # num_tables values

        state.append(obs.current_cost)      # 1 value

        # total: num_tables * 5 + 1
        return torch.tensor(state, dtype=torch.float32)

    # -------- Action encoding --------
    def decode_action(self, action_id):
        table = action_id // 6
        rem = action_id % 6
        join = rem // 2
        index = rem % 2
        return table, join, index

    # -------- Select action --------
    def select_action(self, obs):
        state = self.encode_state(obs)

        if random.random() < self.epsilon:
            action_id = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                action_id = torch.argmax(q_values).item()

        return self.decode_action(action_id), action_id

    # -------- Train step --------
    def train_step(self, state, action, reward, next_state, done):
        q_values = self.forward(state)
        next_q_values = self.forward(next_state)

        target = q_values.clone().detach()
        target[action] = reward + (0 if done else self.gamma * torch.max(next_q_values))

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay