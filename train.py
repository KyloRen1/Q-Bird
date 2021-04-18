import time
from dqn import DQN
from flappy_bird_env import Environment
import torch
import numpy as np
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_episodes', default=5, type=int)
	parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
	parser.add_argument('--fps', default=1, type=int)
	parser.add_argument('--dist_to_pipe', default=50, type=int)
	parser.add_argument('--debug', default=True, type=bool)
	parser.add_argument('--inference', default=False, type=bool)
	parser.add_argument('--dist_between_pipes', default=220, type=int)
	parser.add_argument('--obs_this_pipe', default=True, type=bool)
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	args.device = torch.device(args.device)
	
	env = Environment(draw=True, 
		fps=args.fps, 
		debug=args.debug, 
		dist_to_pipe=args.dist_to_pipe,
		dist_between_pipes=args.dist_between_pipes,
		obs_this_pipe=args.obs_this_pipe)

	agent_model = DQN(env.observation_space.n, env.action_space.n).to(args.device)

	if args.inference:
		agent_model.load_checkpoint()

	best_reward = 0.0
	for i in range(args.n_episodes):
		state = env.reset()
		total_reward = 0.0

		while True:
			state = torch.tensor(np.array([state], copy=False)).float().to(args.device)

			q_values = agent_model(state)
			action = q_values.max(1)[1].view(1, 1)

			state, reward, done, _ = env.step(action)
			total_reward += reward

			if done:
				break

		if total_reward > best_reward and not args.inference:
			agent_model.save_checkpoint()


		print(f'Total reward: {total_reward}')
