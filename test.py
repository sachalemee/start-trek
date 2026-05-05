import numpy as np
import matplotlib.pyplot as plt

n_states = 16 # nombre de pos
n_actions = 4 # nombre d'actons possibles
goal_state = 15 # case

Q_table = np.zeros((n_states, n_actions)) # vide

learning_rate = 0.8 # a quel point tu favorise les nouvelles infos
discount_factor = 0.95 # valuation des récompenses
exploration_prob = 0.2 # exploration
epochs = 1000000 # nombre de tour

def get_next_state(state, action):
    """
    calculate next state with action
    """
    row, col = divmod(state, 4)  

    if action == 0 and col > 0:
        col -= 1
    elif action == 1 and col < 3:      
        col += 1
    elif action == 2 and row > 0:     
        row -= 1
    elif action == 3 and row < 3:     
        row += 1

    return row * 4 + col

for epoch in range(epochs ): # each episode  
    current_state = np.random.randint(0, n_states)  

    while True:
        if np.random.rand() < exploration_prob: # si exploration
            action = np.random.randint(0, n_actions) # rand
        else:
            action = np.argmax(Q_table[current_state]) # get the index of the best action

        next_state = get_next_state(current_state, action)

        reward = 1 if next_state == goal_state else 0

        Q_table[current_state, action] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action]
        )

        if next_state == goal_state: # terminator
            break

        current_state = next_state
