import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define your datasets, partitioners, and algorithms
datasets = ['BloodMNIST']
algorithms = ['fedavg', 'fednova']
partitioners = {
    'iid_poisoned': ['poison_fraction0.8', 'poison_fraction0.8', 'poison_fraction0.8', 'poison_fraction0.8']
}

# Base directory where the CSV results are stored
base_dir = '/home/abdenour/PycharmProjects/fl_with_flower/results'

# Assign distinct colors for each algorithm
algorithm_colors = {
    'fedavg': '#1f77b4',  # Blue
    'fedavgm': '#ff7f0e',  # Orange
}

# Function to load accuracy data from a CSV file
def load_accuracy_data(algorithm, dataset, partitioner, partitioner_type, file_name):
    csv_path = os.path.join(base_dir, algorithm, dataset, partitioner, file_name + '.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"File {csv_path} does not exist.")
        return pd.DataFrame()

# Function to create a single subplot for iid_poisoned partitioner with error bars
def create_single_iid_poisoned_subplot(figure_title, partitioner_type):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(figure_title, fontsize=16)

    dataset = datasets[0]  # Use only 'BloodMNIST' dataset
    partitioner_files = partitioners[partitioner_type]  # Get the files for the 'iid_poisoned' partitioner

    bar_width = 0.35  # Width of each bar
    x_positions = np.arange(len(partitioner_files))  # Position of bars for each partitioner

    for k, algorithm in enumerate(algorithms):
        mean_accuracies = []  # Store the mean accuracies for the algorithm
        std_devs = []  # Store the standard deviation for error bars
        for j, partitioner_file in enumerate(partitioner_files):
            # Load accuracy data for each algorithm-dataset-partitioner combination
            data = load_accuracy_data(algorithm, dataset, partitioner_type, partitioner_type, partitioner_file)

            if not data.empty:
                # Extract the last 10 rounds (tail) and calculate the mean and standard deviation of accuracy
                tail_data = data.tail(40)
                mean_accuracy = tail_data['Accuracy'].mean()
                std_dev = tail_data['Accuracy'].std()  # Standard deviation for error bars
                mean_accuracies.append(mean_accuracy)
                std_devs.append(std_dev)
            else:
                mean_accuracies.append(0)  # Handle the case where there's no data
                std_devs.append(0)  # No error bar if there's no data

        # Create a bar plot for the mean accuracies of this algorithm with error bars
        ax.bar(x_positions + k * bar_width, mean_accuracies, yerr=std_devs, width=bar_width, label=algorithm,
               color=algorithm_colors[algorithm], capsize=5)

    # Set y-axis limits to [0, 1]
    ax.set_ylim([0, 1])

    # Set labels, legend, and title for the subplot
    ax.set_xticks(x_positions + bar_width / 2)
    ax.set_xticklabels(partitioner_files)
    ax.set_xlabel('Partitioner File (Poison Fractions)')
    ax.set_ylabel('Mean Accuracy (Tail 10)')
    ax.legend(title="Algorithm")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(base_dir, figure_title + '.pdf'))
    plt.show()

# Create the figure for iid_poisoned partitioner with error bars
create_single_iid_poisoned_subplot('Figure: IID Poisoned Partitioner Mean Accuracy Comparison with Error Bars', 'iid_poisoned')
