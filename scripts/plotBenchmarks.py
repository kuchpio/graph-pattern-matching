import os
import matplotlib.pyplot as plt

# Directory containing the files
directory = "/home/borys/studia/Beng/graph-pattern-matching/out/build/x64-release-linux/app/EfficiencyTests"

# Function to process a file and generate a plot
def process_file(file_path, file_name):
    small_sizes = []
    times = []
    
    # Read the file and extract data
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.split(';')
                time = float(parts[1].strip().split()[0])  # Extract time in seconds
                small = int(parts[3].split('=')[1].strip())  # Extract small graph size
                small_sizes.append(small)
                times.append(time)
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(small_sizes, times, marker='o', linestyle='-', color='b', label='Time')
    plt.title(f"Performance for {file_name}")
    plt.xlabel("Small Graph Size")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_filename = f"{file_name}_plot.png"
    plt.savefig(output_filename)
    plt.close()

# Iterate through all files in the directory
for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    if os.path.isfile(file_path):  # Check if it's a file
        process_file(file_path, file_name)

print("Plots generated and saved.")