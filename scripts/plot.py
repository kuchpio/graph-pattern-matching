import os
import matplotlib.pyplot as plt

def extract_data(filepath):
    """Extract data from a file. Returns sizes, times, and success statuses."""
    sizes = []
    times = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) < 3:
                continue
            status = parts[0].strip()
            time = float(parts[1].split()[0])  # Extract time in seconds
            size = int(parts[2].split('=')[1]) if 'small' in filepath else int(parts[3].split('=')[1])

            if status == "success":
                sizes.append(size)
                times.append(time)

    return sizes, times

def plot_data(base_dir):
    """Generate plots for all directories."""
    output_dir = os.path.join("./", "benchmarksPlots")
    os.makedirs(output_dir, exist_ok=True)

    for directory in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, directory)
        if not os.path.isdir(dir_path):
            continue

        plt.figure(figsize=(10, 6))
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if not os.path.isfile(file_path):
                continue

            pattern_name = os.path.splitext(file_name)[0]
            is_big_const = "_bigConst" in pattern_name
            pattern_name = pattern_name.split("_bigConst")[0].split("_smallConst")[0]  # Trim to before _bigConst or _smallConst

            sizes, times = extract_data(file_path)

            if sizes and times:
                x_label = "Size of Small Graph" if is_big_const else "Size of Big Graph"
                plt.plot(sizes, times, label=f"{pattern_name} ({x_label})")

        plt.title(f"Performance Analysis - {directory}")
        plt.xlabel("Size of Changing Graph")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{directory}_plot.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")

if __name__ == "__main__":
    base_directory = "../out/build/x64-release-linux/app/EfficiencyTests"  # Adjust this path if needed
    plot_data(base_directory)
