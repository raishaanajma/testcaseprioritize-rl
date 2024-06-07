import json

def calculate_aprc(covered_requirements, total_requirements):
    """
    covered_requirements (list of sets): List of sets, each containing covered requirements for an episode.
    total_requirements (int): Total number of unique requirements.
    """
    aprc_list = []
    for reqs in covered_requirements:
        aprc_list.append(len(reqs) / total_requirements)
    aprc_value = sum(aprc_list) / len(aprc_list) if aprc_list else 0
    aprc_value += 1 / (2 * total_requirements)  # Adding the smoothing term
    return aprc_value

def calculate_total_cost(test_case_costs):
    total_cost = 0
    for cost in test_case_costs.values():
        total_cost += cost
    return total_cost

def calculate_average_cost_per_test_case(total_cost, total_requirements):
    cost_per_test_case = total_cost / total_requirements
    return cost_per_test_case

# Load data from JSON
with open('results_testing.json', 'r') as f:
    data = json.load(f)

covered_requirements = data['covered_requirements']
total_requirements = data['total_requirements']
test_case_costs = data['test_case_costs']

# Calculate APRC value
aprc_value = calculate_aprc(covered_requirements, total_requirements)

# Calculate total cost
total_cost = test_case_costs

# Calculate average cost per test case
average_cost_per_test_case = calculate_average_cost_per_test_case(total_cost, total_requirements)

print(f"\nAverage Percentage of Requirement Coverage (APRC): {aprc_value:.6f} ({aprc_value * 100:.2f}%)")
print(f"Total Cost: ${total_cost}")
print(f"Average Cost per Requirement: ${average_cost_per_test_case:.2f}")