import numpy as np
import matplotlib.pyplot as plt



def gaussian_reward(mean=2, variance=1):
    return np.random.normal(loc=mean, scale=np.sqrt(variance))

def fair_coin_toss():
    return 5 if np.random.rand() < 0.5 else -6

def poisson_reward(lam=2):
    return np.random.poisson(lam=lam)

def exponential_reward(scale=3):
    return np.random.exponential(scale=scale)

def crazy_button():
    reward_functions = [gaussian_reward, fair_coin_toss, poisson_reward, exponential_reward]
    selected_function = np.random.choice(reward_functions)
    return selected_function()



def sub_plot_results(rewards_list, epsilons):
    #plot all the rewards for each epsilon value as sub plots
    for rewards, epsilon in zip(rewards_list, epsilons):
        
        plt.subplot((len(epsilons)//2)+1, 2, epsilons.index(epsilon) + 1)
        plt.plot(rewards, label=f'epsilon = {epsilon}', )
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
    plt.show()



def plot_results(rewards_list, epsilons):
    #plot all the rewards for each epsilon value
    for rewards, epsilon in zip(rewards_list, epsilons):
        
        plt.plot(rewards, label=f'epsilon = {epsilon}',alpha=0.8 )
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
    plt.show()



def pull(arm):
        if arm == 0:
            return gaussian_reward()
        elif arm == 1:
            return fair_coin_toss()
        elif arm == 2:
            return poisson_reward()
        elif arm == 3:
            return exponential_reward()
        elif arm == 4:
            return crazy_button()
    
#----------------------------------------------main code------------------------------------------------------------

n_episodes = 1000
episode_length = 100
epsilons = [0.2, 0.01,0.005,0.9]
rewards_list = []#list to store the rewards for each epsilon value



for epsilon in epsilons:
    
    n_arms = 5
    q_values = np.zeros(n_arms)  # Estimated action values
    action_counts = np.zeros(n_arms)#(denominator value in the formula for updating q values)
    def select_action():
        if np.random.rand() < epsilon:
            return np.random.choice(len(q_values))
        else:
            
            return np.argmax(q_values)

    def update_q_values(arm, reward):
        action_counts[arm] += 1
        q_values[arm] += (reward - q_values[arm]) / action_counts[arm]
    
    
    rewards = np.zeros(n_episodes)
    

    for episode in range(n_episodes):
        total_reward = 0
        for step in range(episode_length):#for each step in the episode(100 steps)
           #select a random action
            action = select_action()
            #do the action get the reward
            reward = pull(action)
            #update the q values
            update_q_values(action, reward)
            #calculate the total reward
            total_reward += reward
        #store the total reward for the episode   
        rewards[episode] = total_reward
    #store the rewards for each epsilon value
    print(action_counts, q_values, rewards)
    rewards_list.append(rewards)


plot_results(rewards_list, epsilons)
sub_plot_results(rewards_list, epsilons)