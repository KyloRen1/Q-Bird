import torch.nn as nn


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
	def __init__(self, input_shape, n_actions, fc1_output=64, fc2_output=64):
		self.fc1 = nn.Linear(input_shape, fc1_output)
		self.fc2 = nn.Linear(fc1_output, fc2_output)
		self.fc3 = nn.Linear(fc2_output, n_actions)

		self.relu = nn.ReLU()

	def forward(self, observation):
		x = self.relu(self.fc1(observation))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)