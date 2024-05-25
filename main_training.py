import seed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

#default cost for missing test cases
DEFAULT_COST_VALUE = 0

class PolicyNetwork(nn.Module): #neural network block
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

class TestCasePrioritizationEnvironment: #environment where agent interacts
    def __init__(self, test_cases, costs, value_priorities, complexities, requirements):
        self.test_cases = test_cases #test case ID
        self.costs = costs #attribute
        self.value_priorities = value_priorities #attribute
        self.complexities = complexities #attribute
        self.requirements = requirements #attribute
        self.state = np.zeros(len(test_cases)) #initial state
        self.total_cost = 0
        self.selected_test_cases_sequences = [] #store selected test cases for each episode
        self.total_rewards = [] #store total rewards for each episode
        self.covered_requirements = [] #store covered requirements for each episode

    def step(self, action):
        action_scalar = action.item() #convert action tensor to scalar
        selected_test_case = self.test_cases[action_scalar] #execute selected test cases
        executed_test_case_cost = self.costs.get(selected_test_case, DEFAULT_COST_VALUE) #get the cost for the selected test case
        self.total_cost += executed_test_case_cost
        #calculate reward based on cost, value priority, and complexity
        reward = (4 - self.value_priorities[selected_test_case]) * self.complexities[selected_test_case] / (executed_test_case_cost)
        #update state
        self.state = np.zeros(len(self.test_cases)) #reset state
        self.state[action_scalar] = 1
        self.selected_test_cases_sequences[-1].append(selected_test_case) #store selected test case for this step
        self.covered_requirements[-1].update({self.requirements[selected_test_case]})
        return self.state, reward, self.total_cost

    def reset(self):
        self.state = np.zeros(len(self.test_cases)) #reset state
        self.total_cost = 0
        self.selected_test_cases_sequences.append([]) #start new episode
        self.total_rewards.append(0) #initialize total reward for new episode
        self.covered_requirements.append(set()) #initialize covered requirements for new episode
        return self.state

#load the dataset
df = pd.read_excel('Test_Project_MIS.xlsx')
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

test_cases = train_df['Test Cases'].tolist()
costs = train_df.set_index('Test Cases')['Cost'].to_dict()
value_priorities = train_df.set_index('Test Cases')['Weights'].to_dict()
complexities = train_df.set_index('Test Cases')['Complexity'].to_dict()
requirements = train_df.set_index('Test Cases')['B_Req'].to_dict()

#modify value priorities
for key, value in value_priorities.items():
    if value == 1:
        value_priorities[key] = 3
    elif value == 3:
        value_priorities[key] = 1

total_requirements = len(set(requirements.values()))

#prepare the environment
env = TestCasePrioritizationEnvironment(test_cases, costs, value_priorities, complexities, requirements)

#define the neural network
input_size = len(test_cases)  #update input_size based on the number of features
hidden_size = 1024  #number of neurons or units
output_size = len(test_cases)
policy_net = PolicyNetwork(input_size, hidden_size, output_size)

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
gamma = 0.95 #discount factor
num_episodes = 10
max_steps_per_episode = len(test_cases)

for episode in range(num_episodes):
    state = env.reset()
    episode_log_probs = []
    episode_rewards = []
    for step in range(max_steps_per_episode): 
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item() #convert tensor to scalar
        action_tensor = torch.tensor([action]) #convert scalar to tensor
        episode_log_probs.append(action_dist.log_prob(action_tensor)) #pass action tensor
        next_state, reward, total_cost = env.step(action_tensor)
        episode_rewards.append(reward)
        env.total_rewards[-1] += reward #accumulate total reward for the episode
        state = next_state
    returns = []
    R = 0
    for r in episode_rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    episode_log_probs = torch.stack(episode_log_probs)
    policy_loss = (-episode_log_probs * returns).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

#find maximum reward
max_reward = max(env.total_rewards)
max_reward_index = env.total_rewards.index(max_reward)
max_reward_sequence = env.selected_test_cases_sequences[max_reward_index]

torch.save(policy_net.state_dict(), 'policy_net.pth') #save trained model

#print sequence of test cases and total reward for each episode
print("Final Result - Sequence of Selected Test Cases and Total Reward for Each Episode:")
for i, (selected_test_cases, total_reward) in enumerate(zip(env.selected_test_cases_sequences, env.total_rewards), start=1):
    print_test_case = selected_test_cases[:5] #print only 5 test cases for each episode
    print(f"Episode {i}: {print_test_case} \n> REWARD: {total_reward}\n")

#print the maximum reward and its sequence
print(f"[TRAINING] MAX Reward: {max_reward}\nEpisode {max_reward_index + 1}: {max_reward_sequence[:5]}")