import os
import re
import matplotlib.pyplot as plt

def extract_data_from_file(file_path):
    """Extracts 'big' and 'time' values from the file."""
    big_values = []
    time_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"success; ([\d.]+) seconds; big=(\d+); small=\d+;", line)
            if match:
                time = float(match.group(1))
                big = int(match.group(2))
                time_values.append(time)
                big_values.append(big)
    
    return big_values, time_values

def plot_data_from_directory(directory_path):
    """Plots 'time' vs 'big' for each file in the directory."""
    plt.figure(figsize=(10, 6))

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.isfile(file_path):
            big_values, time_values = extract_data_from_file(file_path)

            if big_values and time_values:
                plt.plot(big_values, time_values, label=file_name)

    plt.xlabel('search space graph Size')
    plt.ylabel('Time (seconds)')
    plt.title('matching time scaling')
    plt.legend()
    plt.grid(True)
    plt.savefig("comapirsion")

# Example usage:
directory_path = "../out/build/x64-release-linux/app/EfficiencyTests/compare"  # Replace with the path to your directory
plot_data_from_directory(directory_path)
