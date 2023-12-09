import numpy as np
import matplotlib.pyplot as plt
from env import Connect4Env
from node import MCTSNode
from algorithm import mcts
from demo import random_move
from tqdm import tqdm

def evaluate_agent(games=100, mcts_iterations=100):
    outcomes = []  # 1 for MCTS win, -1 for Random win, 0 for draw
    steps_to_win = []  # Record steps only when MCTS wins

    for _ in tqdm(range(games), desc="Games"):
        env = Connect4Env()
        current_player = 'p1'  # MCTS starts
        steps = 0
        winner = None

        while not env.isDone:
            if current_player == 'p1':
                # MCTS agent's turn
                root = MCTSNode(env)
                action = mcts(env, root, current_player, iterations=mcts_iterations)
                _, reward = env.make_move(action, current_player)
                steps += 1
            else:
                # Random agent's turn
                action = random_move(env)
                if action is not None:
                    _, reward = env.make_move(action, current_player)


            if env.isDone:
                if reward == env.reward['win']:
                    winner = current_player
                break

            # Switch players
            current_player = 'p2' if current_player == 'p1' else 'p1'


        if winner == 'p1':  # MCTS won
            outcomes.append(1)
            steps_to_win.append(steps)
        elif winner == 'p2':  # Random agent won
            outcomes.append(-1)
        else:
            outcomes.append(0)  # Draw

    return outcomes, steps_to_win

def make_one_move(env, current_player, mcts_iterations=100):
    root = MCTSNode(env)
    action = mcts(env, root, current_player, iterations=mcts_iterations)

    return action


