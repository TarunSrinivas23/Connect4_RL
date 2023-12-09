import sys
sys.path.append('MCTS/')
sys.path.append('pytorch_dqm/')
from tqdm import tqdm
import time
import numpy as np

from MCTS.evaluate import evaluate_agent, make_one_move


from env import Connect4Env
import random
from pytorch_dqm.agent import select_action
import torch
from pytorch_dqm.model import DQN
#GPU recommended


games_to_play = 100

# Evaluate the MCTS agent VS Random Agent
iterations_per_step = 100
# time_start = time.time()
# outcomes, steps_to_win = evaluate_agent(games=games_to_play, mcts_iterations=iterations_per_step)
# time_end = time.time()
# time_taken = time_end-time_start
# # Calculate win rate and winning steps_average
# win_rates = np.cumsum([1 if outcome == 1 else 0 for outcome in outcomes])
# winning_steps_average = np.sum(steps_to_win) / len(steps_to_win)

# print(f"Win Rate: {win_rates[-1] / games_to_play * 100:.2f}%")
# print(f"Average steps to win by MCTS algorithm: {winning_steps_average:.2f}")
# print(f"Time taken: {time_taken:.2f} seconds, {time_taken/games_to_play:.2f} seconds per game at {iterations_per_step} iterations per step")
# # Evaluate the DQN agent VS Random Agent


env = Connect4Env()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Evaluating on:",device)

n_actions = env.board_width
policy_net = DQN(n_actions).to(device)
if device == torch.device("cuda:0"):
    policy_net.load_state_dict(torch.load("pytorch_dqm/DQN_plainCNN.pth"))
else:
    policy_net.load_state_dict(torch.load("pytorch_dqm/DQN_plainCNN.pth", map_location=torch.device('cpu')))

def random_agent(actions):
    return random.choice(actions)

def evaluate_model():
    steps_to_win = []
    wins =[]
    env.reset()
    time_start = time.time()
    for _ in tqdm(range(games_to_play), desc="Games"):
        move_count = 0
        env.reset()
        # print("Player 1 is AI, Player 2 is random agent")
        while not env.isDone:
            state = env.board_state.copy()
            available_actions = env.get_available_actions()
            action = select_action(policy_net, state, available_actions, device , training=False)
            state, reward = env.make_move(action, 'p1')

            move_count += 1
            if reward == 1:
                steps_to_win.append(move_count)
                wins.append(1)
                break

            available_actions = env.get_available_actions()
            action = random_agent(available_actions)
            state, reward = env.make_move(action, 'p2')

            if reward==1:
                # print("Random agent wins!")
                break

    time_end = time.time()

    return sum(wins)/games_to_play, sum(steps_to_win)/len(steps_to_win), time_end-time_start

win_rate, moves_taken, time_taken = evaluate_model()
print(f"Win Rate: {win_rate * 100:.2f}%")
print(f"Average steps to win by Deep Q Learning algorithm: {moves_taken:.2f}")
print(f"Time taken: {time_taken:.2f} seconds, {time_taken/games_to_play:.2f} seconds per game")


# Evaluate the MCTS agent VS DQN Agent

def evaluate_model(games_to_play, iterations_per_step=10):
    steps_to_win_dqn = [0]
    steps_to_win_mcts = [0]
    wins_dqn =[]
    wins_mcts =[]
    env.reset()
    time_start = time.time()
    for _ in tqdm(range(games_to_play), desc="Games"):
        move_count = 0
        env.reset()

        # print("Player 1 is DQN, Player 2 is MCTS agent")
        while not env.isDone:
            state = env.board_state.copy()
            available_actions = env.get_available_actions()
            if move_count == 0:
                action = random_agent(available_actions) #random agent starts
            else:
                action = select_action(policy_net, state, available_actions, device , training=False)
            # print(action)
            state, reward = env.make_move(action, 'p1')
            # env.render()
            if reward == 1:
                steps_to_win_dqn.append(move_count)
                wins_dqn.append(1)
                # print("DQN Win",env.board_state)
                break

            if move_count == 0:
                action = random_agent(available_actions) #random agent starts
            else:
                action = make_one_move(env, 'p2', mcts_iterations=iterations_per_step)

            # print("MCTS:",action)
            state, reward = env.make_move(action, 'p2')
            # print(reward)
            # print(env.board_state)
            # env.render()
            move_count += 1

            if reward==1:
                # print("MCTS agent wins!")
                steps_to_win_mcts.append(move_count)
                wins_mcts.append(1)
                # print("MCTS Win",env.board_state)
                break
        # env.render()


    time_end = time.time()

    return sum(wins_dqn)/games_to_play, sum(steps_to_win_dqn)/len(steps_to_win_dqn), sum(wins_mcts)/games_to_play, sum(steps_to_win_mcts)/len(steps_to_win_mcts), time_end-time_start

games_to_play = 100
win_rate_dqn, moves_taken_dqn, win_rate_mcts, moves_taken_mcts, time_taken = evaluate_model(games_to_play, iterations_per_step=iterations_per_step)
print(f"Win Rate (DQN): {win_rate_dqn * 100:.2f}%")
print(f"Average steps to win by Deep Q Learning algorithm: {moves_taken_dqn:.2f}")
print(f"Win Rate (MCTS): {win_rate_mcts * 100:.2f}%")
print(f"Average steps to win by MCTS algorithm: {moves_taken_mcts:.2f}")
print(f"Time taken: {time_taken:.2f} seconds, {time_taken/games_to_play:.2f} seconds per game")