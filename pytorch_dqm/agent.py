import random
import torch
import numpy as np
import math

def select_action(policy_net, state, available_actions, device,epsilon_decay = None,steps_done=None, training=True):
    # batch and color channel
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
    epsilon = random.random()
    if training:
        eps_threshold = epsilon_decay['EPS_END'] + (epsilon_decay['EPS_START'] - epsilon_decay['EPS_END']) * math.exp(-1 * steps_done / epsilon_decay['EPS_DECAY'])
    else:
        eps_threshold = 0

    # follow epsilon-greedy policy
    if epsilon > eps_threshold:
        with torch.no_grad():
            # action recommendations from policy net
            r_actions = policy_net(state)[0, :]
            state_action_values = [r_actions[action].cpu() for action in available_actions]
            state_action_values = np.array(state_action_values)
            state_action_values = torch.tensor(state_action_values).detach().cpu().numpy()
            argmax_action = state_action_values.argmax()
            greedy_action = available_actions[argmax_action]
            return greedy_action
    else:
        return random.choice(available_actions)
