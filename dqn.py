import torch.nn as nn
import collections
from collections import namedtuple
import numpy as np
import os
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_output=64, fc2_output=64):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_shape, out_features=fc1_output)
        self.fc2 = nn.Linear(in_features=fc1_output, out_features=fc2_output)
        self.fc3 = nn.Linear(in_features=fc2_output, out_features=n_actions)

        self.relu = nn.ReLU()

        os.makedirs('model_weights', exist_ok=True)
        self.checkpoint_file = os.path.join('model_weights', 'dqn')

    def forward(self, observation):
        x = self.relu(self.fc1(observation))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=True)
        states, actions, next_states, rewards, dones = zip(*[self.memory[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

    def __len__(self):
        return len(self.memory)