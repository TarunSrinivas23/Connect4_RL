import random
from copy import deepcopy

def mcts(env, root, player, iterations=100):
    for _ in range(iterations):
        node = root
        # Selection
        while node.children:
            node = node.best_child()

        # Expansion
        if not node.state.isDone:
            actions = node.state.get_available_actions()
            for action in actions:
                new_env = deepcopy(node.state)  # Create a new instance of the game environment
                new_env.make_move(action, player)
                node.add_child(new_env, action)

            node = random.choice(node.children)

        # Simulation
        if player == 'p1':
            # print("Simulation")
            current_env = deepcopy(node.state)  # Use a copy of the environment for simulation
            while not current_env.isDone:
                possible_moves = current_env.get_available_actions()
                action = random.choice(possible_moves)
                current_env.make_move(action, player)
            break
        else:
            current_env = deepcopy(node.state)
            current_player = 'p1'  # Assume p1 starts the game
            while not current_env.isDone:
                action = random.choice(current_env.get_available_actions())
                current_env.make_move(action, current_player)
                current_player = 'p2' if current_player == 'p1' else 'p1'

        # Backpropagation
        reward = current_env.check_game_done(player)  # Check the game outcome
        while node:
            node.update(reward)
            node = node.parent

    return root.best_child(c_param=0).parent_action

