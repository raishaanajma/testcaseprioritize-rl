import json
import pandas as pd
from sklearn.model_selection import train_test_split

def calculate_aprc(covered_requirements, total_requirements):
    aprc_list = []
    for reqs in covered_requirements:
        aprc_list.append(len(reqs) / total_requirements)
    aprc_value = sum(aprc_list) / len(aprc_list) if aprc_list else 0
    aprc_value += 1 / (2 * total_requirements)  #adding the smoothing term
    return aprc_value

def calculate_total_cost(test_case_costs):
    return sum(test_case_costs.values())

def calculate_average_cost_per_test_case(total_cost, total_requirements):
    return total_cost / total_requirements

def calculate_cost_for_percentage(dataset, percentage):
    dataset_to_consider = int(len(dataset) * percentage)
    return dataset['Cost'].head(dataset_to_consider).sum()

#load data from JSON
with open('results_testing.json', 'r') as f:
    data = json.load(f)

#load the full dataset
df = pd.read_excel('Test_Project_MIS.xlsx')

#split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

covered_requirements = data['covered_requirements']
total_requirements = data['total_requirements']
test_case_costs = data['test_case_costs']

#calculate APRC value
aprc_value = calculate_aprc(covered_requirements, total_requirements)

#calculate cost based on APRC value of the test dataset
percentage_covered = aprc_value
selected_test_cases_cost = calculate_cost_for_percentage(test_df, percentage_covered)
average_cost_per_requirement_aprc = calculate_average_cost_per_test_case(selected_test_cases_cost, total_requirements * percentage_covered)

print(f"\nAverage Percentage of Requirement Coverage (APRC): {aprc_value:.6f} ({aprc_value * 100:.2f}%)")
print(f"Total Cost Based on APRC Value ({percentage_covered * 100:.2f}% of Test Dataset): ${selected_test_cases_cost}")
print(f"Average Cost per Requirement Based on APRC: ${average_cost_per_requirement_aprc:.2f}\n")