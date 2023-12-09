import random
from env import Connect4Env
from node import MCTSNode
from algorithm import mcts

# Function to play a random move for the opponent
def random_move(env):
    available_actions = env.get_available_actions()
    return random.choice(available_actions) if available_actions else None

# Function to play a game
def play_game():
    env = Connect4Env()
    current_player = 'p1'  # MCTS player
    opponent_player = 'p2'  # Random move player

    while not env.isDone:
        print(f"Player {current_player}'s turn")
        if current_player == 'p1':
            # MCTS player's turn
            root = MCTSNode(env)
            action = mcts(env, root, current_player, iterations=100)
            _, reward = env.make_move(action, current_player)
            print(f"MCTS Player's Move: Column {action}")
        else:
            # Opponent's turn
            action = random_move(env)
            if action is not None:
                _, reward = env.make_move(action, opponent_player)
                print(f"Opponent's Move: Column {action}")
            else:
                print("No available actions for the opponent")

        # Switch players
        current_player = 'p2' if current_player == 'p1' else 'p1'
        env.render()

        # Check for game end
        if env.isDone:
            if reward == env.reward['win']:
                print(f"Player {current_player} wins!")
            elif reward == env.reward['draw']:
                print("It's a draw!")
            else:
                print(f"Player {opponent_player} wins!")
            break

# # Play one game
# play_game()
