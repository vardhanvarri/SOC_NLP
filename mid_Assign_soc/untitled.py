import numpy as np
grid_size=4
empty_value=np.zeros((4,4))
reward_pots={
    (0,3): 10,
    (1,1): "-1",
    (3,0): 1,
    (3,2): "-20",
    (2,3): -45
}
rewards=np.zeros((grid_size,grid_size))
for pos,reward in reward_pots.items():
    rewards[pos]=reward
    
# Define the deterministic transition dynamics
def move(position, action):
    x, y = position
    if action == "up" and x > 0:
        return (x - 1, y)
    elif action == "down" and x < grid_size - 1:
        return (x + 1, y)
    elif action == "left" and y > 0:
        return (x, y - 1)
    elif action == "right" and y < grid_size - 1:
        return (x, y + 1)
    else:
        return position  # If the move is not possible, stay in the same position

value_func=np.zeros((4,4))

# Value iteration parameters
gamma = 0.7  # Discount factor
theta = 0.00001  # Threshold for convergence
actions = ["up", "down", "left", "right"]


def eval_valuefunc(state, value_function, rewards, gamma=0.5):
    x, y = state
    actions = ["up", "down", "left", "right"]
    
    new_value = max(
        rewards[x, y] + gamma * value_function[move((x, y), action)]
        for action in actions
    )
    return new_value
def value_iter(value_func,rewards,gamma=0.5,theta=1e-5):
    
    while True:
        delta=0
        new_value_func = np.zeros((grid_size, grid_size))
        
        for x in range(grid_size):
            for y in range(grid_size):
                state=(x,y)
                print('k')
                v = value_func[x, y]
                new_value_func[x,y]=eval_valuefunc(state, value_func, rewards, gamma=0.5)
                delta = max(delta, abs(v - new_value_func[x, y]))
        value_func = new_value_func.copy()

        if delta < theta:
            break
    
    return value_func
    
optimal_value_func = value_iter(empty_value, rewards, gamma, theta)

# Print the optimal value function
print("Optimal Value Function:")
print(optimal_value_func)