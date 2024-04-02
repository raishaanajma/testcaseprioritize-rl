import numpy as np
import pandas as pd

class TestCasePrioritizationEnvironment:
    def __init__(self, test_cases, costs, value_priorities, historical_success_rates):
        self.test_cases = test_cases
        self.costs = costs
        self.value_priorities = value_priorities
        self.historical_success_rates = historical_success_rates
        self.state = np.zeros(len(test_cases))  # Initial state
        self.total_cost = 0
        self.selected_test_cases_sequence = []  # Store selected test cases for each episode

    def step(self, action):
        # Execute selected test cases
        selected_test_cases = [self.test_cases[i] for i in range(len(self.test_cases)) if action[i] == 1]
        executed_test_cases_cost = sum(self.costs[test_case] for test_case in selected_test_cases)
        self.total_cost += executed_test_cases_cost
        
        # Calculate reward based on value priority and historical success rate
        reward = sum(self.value_priorities[test_case] * self.historical_success_rates[test_case]
                     for test_case in selected_test_cases)
        
        # Update state
        self.state = np.zeros(len(self.test_cases))  # Reset state
        for i in range(len(self.test_cases)):
            if self.test_cases[i] in selected_test_cases:
                self.state[i] = 1
        
        # Store selected test cases for this episode
        self.selected_test_cases_sequence.append(selected_test_cases)
        
        return self.state, reward, self.total_cost

    def reset(self):
        self.state = np.zeros(len(self.test_cases))  # Reset state
        self.total_cost = 0
        return self.state

# Usage
df = pd.read_excel('data_input.xlsx')
test_cases = df['Test Cases'].tolist()
costs = df.set_index('Test Cases')['Cost'].to_dict()
value_priorities = df.set_index('Test Cases')['Value Priorities'].to_dict()
historical_success_rates = df.set_index('Test Cases')['Historical Success Rate'].to_dict()

env = TestCasePrioritizationEnvironment(test_cases, costs, value_priorities, historical_success_rates)

# RL training loop
num_episodes = 100
max_steps_per_episode = 100  # Define maximum number of steps per episode
for episode in range(num_episodes):
    state = env.reset()
    steps = 0
    done = False
    while not done and steps < max_steps_per_episode:
        # Choose action based on RL policy
        action = np.random.randint(2, size=len(test_cases))  # Random action for illustration
        next_state, reward, total_cost = env.step(action)
        print("Episode:", episode + 1, "| Action:", action, "| Reward:", reward, "| Total Cost:", total_cost)
        # Update RL model parameters based on experience
        steps += 1

# Print sequence of selected test cases
print("Final Result - Sequence of Selected Test Cases:")
for i, selected_test_cases in enumerate(env.selected_test_cases_sequence, start=1):
    print("Episode", i, ":", selected_test_cases)
