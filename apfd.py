#calculate APFD

def calculate_apfd(test_cases_sequence, total_faults, faults_matrix):
    """
    Parameters:
    test_cases_sequence (list): The sequence of selected test cases.
    total_faults (int): The total number of faults in the system.
    faults_matrix (dict): A dictionary where keys are test case IDs and values are lists of detected faults.

    """
    # n is the total number of test cases in the sequence
    n = len(test_cases_sequence)
    # m is the total number of faults in the system
    m = total_faults
    # Create a list to store the position of the first detection of each fault
    fault_positions = [0] * m

    # Loop through each test case in the sequence
    for i, test_case in enumerate(test_cases_sequence):
        # Get the list of faults detected by this test case
        detected_faults = faults_matrix.get(test_case, [])
        # Loop through each detected fault
        for fault in detected_faults:
            # If this fault has not been detected before, update its position
            if fault_positions[fault] == 0:
                fault_positions[fault] = i + 1

    # Calculate the sum of the positions where faults were first detected
    sum_fault_positions = sum(fault_positions)
    # Calculate the APFD value using the formula
    apfd = 1 - (sum_fault_positions / (n * m)) + (1 / (2 * n))
    return apfd

# Example usage:
if __name__ == "__main__":
    # Define the sequence of test cases that gave the maximum reward
    max_reward_sequence = ['TC_821', 'TC_509', 'TC_328', 'TC_894', 'TC_1248']

    # Define the total number of faults in the system
    total_faults = 10
    # Define a dictionary that maps each test case to the faults it detects
    faults_matrix = {
        'TC_821': [0, 1, 2],
        'TC_509': [2, 3],
        'TC_328': [0, 4],
        'TC_894': [1, 5],
        'TC_1248': [6, 7, 8, 9]
    }

    # Calculate the APFD value for the given sequence of test cases
    apfd_value = calculate_apfd(max_reward_sequence, total_faults, faults_matrix)
    # Print the APFD value
    print(f"APFD Value: {apfd_value}")