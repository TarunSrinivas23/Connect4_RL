import numpy as np

class MCTSNode:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, child_state, action):
        child_node = MCTSNode(child_state, self, action)
        self.children.append(child_node)

    def update(self, reward):
        self.visits += 1
        self.value += (reward - self.value) / self.visits

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_available_actions())

    def best_child(self, c_param=1.4):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child  # Automatically select an unvisited child

            # UCB1 formula
            score = child.value / child.visits + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))

            if score > best_score:
                best_score = score
                best_child = child

        return best_child
