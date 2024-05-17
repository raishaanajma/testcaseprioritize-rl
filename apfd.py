import json

def calculate_apfd(test_cases_sequence, total_faults, faults_matrix):

    """
    test_cases_sequence (list): The sequence of selected test cases.
    total_faults (int): The total number of faults in the system.
    faults_matrix (dict): A dictionary where keys are test case IDs and values are lists of detected faults.
    """

    n = len(test_cases_sequence)
    m = total_faults
    fault_positions = [0] * m

    unique_faults = set(fault for faults in faults_matrix.values() for fault in faults) #create a mapping from fault IDs
    fault_id_to_index = {fault_id: index for index, fault_id in enumerate(unique_faults)}

    for i, test_case in enumerate(test_cases_sequence):
        detected_faults = faults_matrix.get(test_case, [])
        for fault in detected_faults:
            fault_index = fault_id_to_index[fault]
            if fault_positions[fault_index] == 0:
                fault_positions[fault_index] = i + 1

    sum_fault_positions = sum(fault_positions)
    apfd = 1 - (sum_fault_positions / (n * m)) + (1 / (2 * n))
    return apfd

#load data from the JSON file
with open('results.json', 'r') as f:
    data = json.load(f)

max_reward_sequence = data['max_reward_sequence']
total_faults = data['total_faults']
faults_matrix = data['faults_matrix']

apfd_value = calculate_apfd(max_reward_sequence, total_faults, faults_matrix) #calculate APFD value

print(f"APFD Value: {apfd_value}")