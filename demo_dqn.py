from env import Connect4Env
import random
from pytorch_dqm.agent import select_action
import torch
from pytorch_dqm.model import DQN

env = Connect4Env()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_actions = env.board_width
policy_net = DQN(n_actions).to(device)
policy_net.load_state_dict(torch.load("pytorch_dqm/DQN_plainCNN.pth", map_location=torch.device('cpu')))

def random_agent(actions):
    return random.choice(actions)

def demo():
    env.reset()
    print("Player 1 is AI, Player 2 is random agent")
    move_count = 0
    while not env.isDone:
        state = env.board_state.copy()
        available_actions = env.get_available_actions()
        action = select_action(policy_net, state, available_actions, device , training=False)
        state, reward = env.make_move(action, 'p1')
        if move_count == 0:
            pass
        else:
            env.render()
        move_count += 1
        if reward == 1:
            print("AI WINS! Surrender, human!")
            print("Moves taken: ", move_count)
            break

        available_actions = env.get_available_actions()
        action = random_agent(available_actions)
        state, reward = env.make_move(action, 'p2')
        env.render()



demo()