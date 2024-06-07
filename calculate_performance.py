import json

def calculate_aprc(covered_requirements, total_requirements):
    """
    covered_requirements (list of sets): List of sets, each containing covered requirements for an episode.
    total_requirements (int): Total number of unique requirements.
    """
    aprc_list = []
    for reqs in covered_requirements:
        aprc_list.append(len(reqs) / total_requirements)
    return sum(aprc_list) / len(aprc_list) if aprc_list else 0

#load data from JSON
with open('results_testing.json', 'r') as f:
    data = json.load(f)

covered_requirements = data['max_reward_sequence']
total_requirements = data['total_requirements']

#calculate APRC value
aprc_value = calculate_aprc(covered_requirements, total_requirements)

print(f"Average Percentage of Requirement Coverage (APRC): {aprc_value:.6f} ({aprc_value * 100:.2f}%)\n")