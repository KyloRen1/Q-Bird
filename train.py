import time
from dqn import DQN
from environments import Environment
import torch
import numpy as np

#def parse_arguments():
#	parser.add_argument('--')


if __name__ == '__main__':
	env = Environment(fps=1, debug=True, dist_to_pipe=50)

	print(env.observation_space.n, env.action_space.n)

	agent_model = DQN(env.observation_space.n, env.action_space.n)

	best_reward = 0.0
	#for i in range(args.n_episodes):
	for i in range(5):
		state = env.reset()
		print(state)
		total_reward = 0.0

		while True:
			state = torch.tensor(np.array([state], copy=False)).float()

			q_values = agent_model(state)
			action = q_values.max(1)[1].view(1, 1)

			state, reward, done, _ = env.step(action)
			total_reward += reward

			if done:
				break

		if total_reward > best_reward:
			agent_model.save_checkpoint()


		print(f'Total reward: {total_reward}')