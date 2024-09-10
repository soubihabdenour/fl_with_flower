import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define your datasets, partitioners, and algorithms
datasets = ['BloodMNIST', 'PathMNIST', 'TissueMNIST']
algorithms = ['fedavg', 'fedavgm', 'fedprox', 'fedbn', 'fednova']
partitioners = {
    'drichlet': ['alpha0.1', 'alpha0.3', 'alpha0.9'],
    'pathological': ['classes02', 'classes04', 'classes07']
}

# Base directory where the CSV results are stored
base_dir = '/home/abdenour/PycharmProjects/fl_with_flower/results'

# Assign distinct colors for each algorithm
algorithm_colors = {
    'fedavg': '#1f77b4',  # Blue
    'fedavgm': '#ff7f0e',  # Orange
    'fedprox': '#2ca02c',  # Green
    'fedbn': '#d62728',    # Red
    'fednova': '#9467bd'   # Purple
}

# Function to load accuracy data from a CSV file
def load_accuracy_data(algorithm, dataset, partitioner, partitioner_type, file_name):
    csv_path = os.path.join(base_dir, algorithm, dataset, partitioner, file_name + '.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"File {csv_path} does not exist.")
        return pd.DataFrame()

# Function to create 3x3 subplots for given algorithms, datasets, and partitioners (Bar Plot for Mean Accuracy of Tail 10 Rounds)
def create_figure(figure_title, partitioner_type):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(figure_title, fontsize=16)

    for i, dataset in enumerate(datasets):  # Row for each dataset
        partitioner_files = partitioners[partitioner_type]  # Get the files for the current partitioner
        for j, partitioner_file in enumerate(partitioner_files):  # Column for each partitioner
            ax = axs[i, j]
            bar_width = 0.7  # Width of each bar
            x_positions = np.arange(len(algorithms))  # Position of bars for each algorithm

            mean_accuracies = []  # Store the mean accuracies for each algorithm
            for k, algorithm in enumerate(algorithms):
                # Load accuracy data for each algorithm-dataset-partitioner combination
                data = load_accuracy_data(algorithm, dataset, partitioner_type, partitioner_type, partitioner_file)

                if not data.empty:
                    # Extract the last 10 rounds (tail) and calculate the mean accuracy
                    tail_data = data.tail(10)
                    mean_accuracy = tail_data['Accuracy'].mean()
                    mean_accuracies.append(mean_accuracy)
                else:
                    mean_accuracies.append(0)  # Handle the case where there's no data

            # Create a bar plot for the mean accuracies of all algorithms, with distinct colors for each algorithm
            ax.bar(x_positions, mean_accuracies, width=bar_width, color=[algorithm_colors[alg] for alg in algorithms], tick_label=algorithms)

            # Set y-axis limits to [0, 1]
            ax.set_ylim([0, 1])

            # Set labels and titles for each subplot
            if partitioner_type == 'drichlet':
                title = f"Dirichlet - Alpha {partitioner_file.replace('alpha', '').replace('csv', '')}"
            elif partitioner_type == 'pathological':
                title = f"Pathological - Classes {partitioner_file.replace('classes', '')}"

            ax.set_title(f"{title} - {dataset}")
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Mean Accuracy (Tail 10)')
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(base_dir, figure_title + '.pdf'))
    plt.show()


# Create two figures: one for Dirichlet partitioner and one for Pathological partitioner (Bar plot of Mean Tail 10)
create_figure('Figure 1: Dirichlet Partitioner Mean Accuracy Comparisons (Bar Plot Tail 10)', 'drichlet')
create_figure('Figure 2: Pathological Partitioner Mean Accuracy Comparisons (Bar Plot Tail 10)', 'pathological')
