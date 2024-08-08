import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Define the environment and parameters
actions = [0, 1, 2, 3]  # up, down, left, right
rewards = np.array([
    [0, 0, -20, 10],
    [0, -1, 0, 0],
    [-2, 0, 0, -45],
    [1, 0, 0, 20]
])
map_size = 4
num_states = map_size * map_size
num_actions = len(actions)
num_episodes = 1000
gamma = 0.99
alpha = 0.2

# Initialize Q-table
qtable = np.zeros((num_states, num_actions))

# Define the move function
def move(state, action):
    row, col = state
    if action == 0:  # left
        col = max(col - 1, 0)
    elif action == 1:  # down
        row = min(row + 1, map_size - 1)
    elif action == 2:  # right
        col = min(col + 1, map_size - 1)
    elif action == 3:  # up
        row = max(row - 1, 0)
    return (row, col)

# Run the Q-learning algorithm
for _ in range(num_episodes):
    state = (0, 0)
    episode = []
    
    while True:
        state_idx = state[0] * map_size + state[1]
        action = random.choice(actions)
        next_state = move(state, action)
        reward = rewards[next_state]
        next_state_idx = next_state[0] * map_size + next_state[1]
        
        # Update Q-table
        best_next_action = np.argmax(qtable[next_state_idx])
        qtable[state_idx, action] += alpha * (reward + gamma * qtable[next_state_idx, best_next_action] - qtable[state_idx, action])
        
        episode.append((state, action, reward))
        state = next_state
        if state == (3, 3):
            break

def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the policy
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
        ax=ax
    ).set(title="Learned Q-values\nArrows represent best action")
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    plt.show()

# Plot the Q-values and the policy
plot_q_values_map(qtable, map_size)
