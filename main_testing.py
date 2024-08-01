import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
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
train_df, test_df = train_test_split(df, test_size=0.2, random_state=21)

test_cases = test_df['Test Cases'].tolist()
costs = test_df.set_index('Test Cases')['Cost'].to_dict()
value_priorities = test_df.set_index('Test Cases')['Weights'].to_dict()
complexities = test_df.set_index('Test Cases')['Complexity'].to_dict()
requirements = test_df.set_index('Test Cases')['B_Req'].to_dict()

#modify value priorities
for key, value in value_priorities.items():
    if value == 1:
        value_priorities[key] = 3
    elif value == 3:
        value_priorities[key] = 1

total_requirements = len(set(requirements.values()))
env = TestCasePrioritizationEnvironment(test_cases, costs, value_priorities, complexities, requirements) #prepare the environment

#define the neural network
input_size = len(test_cases)
hidden_size = 1024 #number of neurons or units
output_size = len(test_cases)
policy_net = PolicyNetwork(input_size, hidden_size, output_size)

#load the trained model
policy_net.load_state_dict(torch.load('policy_net.pth'))
policy_net.eval()  #set the model to evaluation mode

#evaluation loop
num_episodes = 10 #number of episodes for testing
max_steps_per_episode = len(test_cases)

for episode in range(num_episodes):
    state = env.reset()
    episode_rewards = []

    for step in range(max_steps_per_episode):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():  #no gradient calculation needed during testing
            action_probs = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()

        action_tensor = torch.tensor([action])
        next_state, reward, total_cost = env.step(action_tensor)
        episode_rewards.append(reward)
        state = next_state

    #record the total reward for the episode
    total_episode_reward = sum(episode_rewards)
    env.total_rewards[-1] += total_episode_reward  #accumulate total reward for the episode

#find maximum reward
max_reward = max(env.total_rewards)
max_reward_index = env.total_rewards.index(max_reward)
max_reward_sequence = env.selected_test_cases_sequences[max_reward_index]

#find the cost of maximum reward sequence
max_reward_cost = sum(costs[test_case] for test_case in max_reward_sequence)

#calculate requirement coverage
requirement_coverage = [len(reqs) / total_requirements for reqs in env.covered_requirements]

#calculate APRC value
def calculate_aprc(covered_requirements, total_requirements):
    aprc_list = []
    for reqs in covered_requirements:
        aprc_list.append(len(reqs) / total_requirements)
    aprc_value = sum(aprc_list) / len(aprc_list) if aprc_list else 0
    aprc_value += 1 / (2 * total_requirements)  #adding the smoothing term
    return aprc_value

aprc_value = calculate_aprc(env.covered_requirements, total_requirements)

#calculate cost based on APRC value of the test dataset
def calculate_total_cost(test_case_costs):
    return sum(test_case_costs.values())

def calculate_average_cost_per_test_case(total_cost, total_requirements):
    return total_cost / total_requirements

def calculate_cost_for_percentage(dataset, percentage):
    dataset_to_consider = int(len(dataset) * percentage)
    return dataset['Cost'].head(dataset_to_consider).sum()

percentage_covered = aprc_value
selected_test_cases_cost = calculate_cost_for_percentage(test_df, percentage_covered)
average_cost_per_requirement_aprc = calculate_average_cost_per_test_case(selected_test_cases_cost, total_requirements * percentage_covered)

#save testing result to JSON
data_to_save = {
    "max_reward_sequence": max_reward_sequence,
    "test_case_costs": max_reward_cost,
    "covered_requirements": [list(reqs) for reqs in env.covered_requirements],
    "priorities": value_priorities,
    "total_requirements": total_requirements,
    "selected_test_cases_cost": selected_test_cases_cost,
}

with open('results_testing.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

#print the sequence of test cases and total reward for each episode
print("Final Result - Sequence of Selected Test Cases and Total Reward for Each Episode:")
for i, (selected_test_cases, total_reward) in enumerate(zip(env.selected_test_cases_sequences, env.total_rewards), start=1):
    print_test_case = selected_test_cases[:5]  #print only 5 test cases for each episode
    print(f"Episode {i}: {print_test_case} \n> REWARD: {total_reward}\n")

#print the maximum reward and its sequence
print(f"[TESTING] MAX Reward: {max_reward}\nEpisode {max_reward_index + 1}: {max_reward_sequence[:5]}")