import time
from dqn import DQN
from flappy_bird_env import Environment
import torch
import numpy as np
import argparse
from torch import nn

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--gpu', default=False, type=bool)
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--dist_to_pipe', default=50, type=int)
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--dist_between_pipes', default=220, type=int)
    parser.add_argument('--obs_this_pipe', default=False, type=bool)
    parser.add_argument('--runs', default=10, type=int)
    return parser.parse_args()

def play(args):
    device = torch.device("cuda" if args.gpu else "cpu")
    env = Environment(draw=True, 
        fps=args.fps, 
        debug=args.debug, 
        dist_to_pipe=args.dist_to_pipe,
        dist_between_pipes=args.dist_between_pipes,
        obs_this_pipe=args.obs_this_pipe)

    observation_space = env.get_observation_size_buffer()
    action_space = env.get_action_size()

    network = DQN(observation_space, action_space)
    network.load_checkpoint(args.checkpoint)

    for _ in range(args.runs):
        state = env.reset()
        total_reward = 0.0
        while True:
            state_v = torch.tensor(np.array([state], copy=False)).to(device)
            q_vals_v = network(state_v.float())
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                print("REWARD: ", total_reward)
                break

if __name__ == '__main__':
    args = parse_arguments()
    play(args)
