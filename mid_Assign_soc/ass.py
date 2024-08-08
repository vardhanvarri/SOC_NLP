import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Discretization function
def discretize_state(state, bins):
    position, velocity = state
    position_bin = np.digitize(position, bins['position']) - 1
    velocity_bin = np.digitize(velocity, bins['velocity']) - 1
    return position_bin, velocity_bin

# Create bins for discretization
def create_bins(num_bins):
    position_bins = np.linspace(-1.2, 0.6, num_bins)
    velocity_bins = np.linspace(-0.07, 0.07, num_bins)
    return {'position': position_bins, 'velocity': velocity_bins}


def policy_eval(env, policy, value_function, bins, gamma):
    value_function = np.zeros((len(bins['position']), len(bins['velocity'])))
    while True:
        delta = 0
        for position_bin in range(len(bins['position'])-1):
            for velocity_bin in range(len(bins['velocity'])-1):
                state = (position_bin, velocity_bin)
                v = 0
                for action in range(env.action_space.n):
                    state=env.reset()
                    next_state,reward,terminated,truncated,info=env.step(action)
                    next_position_bin, next_velocity_bin = discretize_state(next_state, bins)
                    print(next_position_bin,next_velocity_bin)

                    if next_position_bin >= len(bins['position']) - 1:
                        next_position_bin = len(bins['position']) - 2
                    if next_velocity_bin >= len(bins['velocity']) - 1:
                        next_velocity_bin = len(bins['velocity']) - 2
                    print(next_position_bin,next_velocity_bin,action,state)
                    v += policy[state][action] * (reward + gamma * value_function[next_position_bin,next_velocity_bin])
                value_function[state] = v
                delta = max(delta, abs(v - value_function[state]))
        if delta < 1e-5:
            break
    return value_function


def policy_improvement(env,value_function,policy,bins,gamma):
    
    policy_stable = False
    while not policy_stable:
        policy_stable = True
           
        for position_bin in range(len(bins['position']) - 1):
            for velocity_bin in range(len(bins['velocity']) - 1):
                state = (position_bin, velocity_bin)
                old_action = np.argmax(policy[state])
                action_values = []
                for action in range(env.action_space.n):
                    env.reset()
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_position_bin, next_velocity_bin = discretize_state(next_state, bins)
                    action_values.append(reward + gamma * value_function[next_position_bin][next_velocity_bin])
                new_action = np.argmax(action_values)
                policy[state] = np.eye(env.action_space.n)[new_action]
                if old_action != new_action:
                    policy_stable = False
    return policy,policy_stable 


def policy_iteration(env, bins, gamma=0.9):
    # Initialize value function and policy
    value_function = np.zeros((len(bins['position']) - 1, len(bins['velocity']) - 1))
    policy = np.ones((len(bins['position']) - 1, len(bins['velocity']) - 1, env.action_space.n)) / env.action_space.n

    while True:
        # Policy Evaluation
        value_function = policy_eval(env, policy, value_function, bins, gamma)
        
        # Policy Improvement
        policy, policy_stable = policy_improvement(env, value_function, policy, bins, gamma)
        
        if policy_stable:
            break
    
    return policy, value_function
    
# Main function
def main():
    env = gym.make('MountainCar-v0')
    num_bins = 20
    bins = create_bins(num_bins)
    
    gamma = 0.99  # Discount factor
    policy, value_function = policy_iteration(env, bins, gamma)
    num_episodes = 100
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        state_discrete = discretize_state(state, bins)
        total_reward = 0

        while True:
            action = np.argmax(policy[state_discrete])
            state, reward, d,one, _ = env.step(action)
            state_discrete = discretize_state(state, bins)
            total_reward += reward
            print(reward ,d,one)
            
            if d or one:
                break

        rewards.append(total_reward)

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

if __name__ == "__main__":
    main()