import torch
import numpy as np

import torch.optim as optim
import random
import matplotlib.pyplot as plt

from model import DQN, optimize_model
from env import Connect4Env
from agent import select_action
from utils import ReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

num_episodes = 50000

epsilon_decay={'EPS_START':0.9,'EPS_END':0.05,'EPS_DECAY':2000}
# epilson decay graph
EPS_START = epsilon_decay['EPS_START']
EPS_END = epsilon_decay['EPS_END']
EPS_DECAY = epsilon_decay['EPS_DECAY']

BATCH_SIZE = 256
GAMMA = 0.999

env = Connect4Env()
memory = ReplayMemory()


steps_done = np.arange(num_episodes)
eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1 * steps_done / EPS_DECAY)
plt.plot(steps_done, eps)
plt.title('Epsilon decay graph')
plt.xlabel('Episode no.')
plt.ylabel('Epsilon')
plt.savefig("Epsilon_Decay.png")


# get max no. of actions from action space
n_actions = env.board_width

height = env.board_height
width = env.board_width

policy_net = DQN(n_actions).to(device)
# target_net will be updated every n episodes to tell policy_net a better estimate of how far off from convergence
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
# set target_net in testing mode
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())


# random agent
def random_agent(actions):
    return random.choice(actions)

# win rate test
def win_rate_test():
    win_moves_taken_list = []
    win = []
    for i in range(100):
        env.reset()
        win_moves_taken = 0

        while not env.isDone:
            state = env.board_state.copy()
            available_actions = env.get_available_actions()
            action = select_action(policy_net,state, available_actions,device, epsilon_decay, training=False)
            state, reward = env.make_move(action, 'p1')
            win_moves_taken += 1

            if reward == 1:
                win_moves_taken_list.append(win_moves_taken)
                win.append(1)
                break

            available_actions = env.get_available_actions()
            action = random_agent(available_actions)
            state, reward = env.make_move(action, 'p2')

    return sum(win)/100, sum(win_moves_taken_list)/len(win_moves_taken_list)

steps_done = 0
training_history = []

from itertools import count

# control how lagged is target network by updating every n episodes
TARGET_UPDATE = 10

for i in range(num_episodes):
    env.reset()
    state_p1 = env.board_state.copy()

    # record every 20 epochs
    if i % 20 == 19:
        win_rate, moves_taken = win_rate_test()
        training_history.append([i + 1, win_rate, moves_taken])
        th = np.array(training_history)
        # print training message every 200 epochs
        if i % 200 == 199:
            print('Episode {}: | win_rate: {} | moves_taken: {}'.format(i, th[-1, 1], th[-1, 2]))

    for t in count():
        available_actions = env.get_available_actions()
        action_p1 = select_action(policy_net, state_p1, available_actions,device, epsilon_decay,  steps_done)
        steps_done += 1
        state_p1_, reward_p1 = env.make_move(action_p1, 'p1')

        if env.isDone:
            if reward_p1 == 1:
                # reward p1 for p1's win
                memory.dump([state_p1, action_p1, 1, None])
            else:
                # state action value tuple for a draw
                memory.dump([state_p1, action_p1, 0.5, None])
            break

        available_actions = env.get_available_actions()
        action_p2 = random_agent(available_actions)
        state_p2_, reward_p2 = env.make_move(action_p2, 'p2')

        if env.isDone:
            if reward_p2 == 1:
                # punish p1 for (random agent) p2's win
                memory.dump([state_p1, action_p1, -1, None])
            else:
                # state action value tuple for a draw
                memory.dump([state_p1, action_p1, 0.5, None])
            break

        # punish for taking too long to win
        memory.dump([state_p1, action_p1, -0.05, state_p2_])
        state_p1 = state_p2_

        # Perform one step of the optimization (on the policy network)
        optimize_model(policy_net, target_net, optimizer, memory, device, BATCH_SIZE, GAMMA)

    # update the target network, copying all weights and biases in DQN
    if i % TARGET_UPDATE == TARGET_UPDATE - 1:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete, Saving')

path = 'DQN_plainCNN.pth'
torch.save(policy_net.state_dict(), path)

plt.figure(figsize=(20,20))
plt.plot(th[:, 0], th[:, 1], c='c')
win_rate_moving_average = np.array([[(i + 19) * 20, np.mean(th[i: i + 20, 1])] for i in range(len(th) - 19)])
plt.plot(win_rate_moving_average[:, 0], win_rate_moving_average[:, 1], c='b', label='moving average of win rate')
plt.legend()
plt.title('Playing against random agent')
plt.xlabel('Episode no.')
plt.ylabel('Win rate')
plt.savefig("Win_rate.png")

plt.figure(figsize=(20,20))
plt.plot(th[:, 0], th[:, 2], c='c')
win_steps_taken_moving_average = np.array([[(i + 19) * 20, np.mean(th[i: i + 20, 2])] for i in range(len(th) - 19)])
plt.plot(win_steps_taken_moving_average[:, 0], win_steps_taken_moving_average[:, 1], c='b', label='moving average of win steps taken')
plt.legend()
plt.xlabel('Episode no.')
plt.ylabel('Average steps taken for a win')
plt.savefig("Avg_steps.png")