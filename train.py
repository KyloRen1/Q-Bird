import time
from dqn import DQN, ReplayMemory
from flappy_bird_env import Environment
import torch
import numpy as np
import argparse
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=False, type=bool)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--dist_to_pipe', default=50, type=int)
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--inference', default=False, type=bool)
    parser.add_argument('--dist_between_pipes', default=220, type=int)
    parser.add_argument('--obs_this_pipe', default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--final_eps', default=0.02, type=float)
    parser.add_argument('--start_eps', default=1.0, type=float)
    parser.add_argument('--eps_decay_final_step', default=1e6, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--target_update_iterations', default=1000, type=float)
    parser.add_argument('--replay_capacity', default=10000, type=int)
    parser.add_argument('--replay_start_step', default=10000, type=int)
    parser.add_argument('--goal_reward', default=250000, type=float)
    return parser.parse_args()

def calculate_loss(batch, policy_model, target_model, gamma, device='cpu'):
    states, actions, rewards, dones, next_states = batch

    states_ = torch.tensor(states, dtype=torch.float32).to(device)
    next_states_ = torch.tensor(next_states, dtype=torch.float32).to(device)
    actions_ = torch.tensor(actions).to(device)
    rewards_ = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    state_action_values = policy_model(states_).gather(1, actions_.unsqueeze(-1)).squeeze(-1)
    next_state_values = target_model(next_states_).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def train(args):
    device = torch.device("cuda" if args.gpu else "cpu")
    env = Environment(draw=False, 
        fps=args.fps, 
        debug=args.debug, 
        dist_to_pipe=args.dist_to_pipe,
        dist_between_pipes=args.dist_between_pipes,
        obs_this_pipe=args.obs_this_pipe)

    observation_space = env.get_observation_size_buffer()
    action_space = env.get_action_size()

    policy_network = DQN(observation_space, action_space).to(device)
    target_network = DQN(observation_space, action_space).to(device)

    optimizer = torch.optim.Adam(policy_network.parameters(), lr=args.lr)

    replay_buffer = ReplayMemory(args.replay_capacity)
    writer = SummaryWriter()

    if args.inference:
        target_network.load_checkpoint()

    best_reward = None
    iteration = 0
    total_reward = 0.0
    rewards = []
    state = env.reset()
    while True:
        epsilon = max(args.final_eps,
                      args.start_eps - iteration / args.eps_decay_final_step)
        
        iteration += 1
        episode_reward = None
        if np.random.rand() < epsilon:
            action = env.get_action_random()
        else:
            state_v = torch.tensor(np.array([state], copy=False)).to(device)
            q_vals_v = policy_network(state_v.float())
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        next_state, reward, done = env.step(action)
        total_reward += reward

        replay_buffer.push(state, action, next_state, reward, done)

        state = next_state

        if done:
            episode_reward = total_reward
            state = env.reset()
            total_reward = 0.0

        if episode_reward is not None:
            rewards.append(episode_reward)
            mean_reward = np.mean(rewards[-80:])
            print(f"Episode {iteration}:  eps {epsilon}  mean reward {mean_reward}  episode reward {episode_reward}")

            writer.add_scalar("epsilon", epsilon, iteration)
            writer.add_scalar("mean_reward", mean_reward, iteration)
            writer.add_scalar("reward", episode_reward, iteration)

            if best_reward is None or best_reward < mean_reward:
                torch.save(policy_network.state_dict(), f"./models/checkpoint_{iteration}")
                print(f"New best reward found: {best_reward} -> {mean_reward}")
                best_reward = mean_reward
            if mean_reward > args.goal_reward:
                print(f"Achieved in {iteration} steps.")
                break

        if len(replay_buffer) < args.replay_start_step:
            continue

        if iteration % args.target_update_iterations == 0:
            target_network.load_state_dict(policy_network.state_dict())

        optimizer.zero_grad()

        batch = replay_buffer.sample(args.batch_size)
        loss = calculate_loss(batch, policy_network, target_network, args.gamma, device=device)

        loss.backward()
        optimizer.step()
    writer.close()



if __name__ == '__main__':
    args = parse_arguments()

    train(args)
    

