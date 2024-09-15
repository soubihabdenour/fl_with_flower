import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define your datasets, partitioners, and algorithms
datasets = ['BloodMNIST', 'poisonedBloodMNIST', 'TissueMNIST']
algorithms = ['fedavg', 'fedavgm', 'fedprox', 'fednova']
partitioners = {
    'drichlet': ['alpha0.9', 'alpha0.3', 'alpha0.1'],
    'pathological': ['classes07', 'classes04', 'classes02']
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

    for i, dataset in enumerate(datasets):  # Row for each dataset
        partitioner_files = partitioners[partitioner_type]  # Get the files for the current partitioner
        for j, partitioner_file in enumerate(partitioner_files):  # Column for each partitioner
            ax = axs[i, j]
            bar_width = 0.5  # Width of each bar
            x_positions = np.arange(len(algorithms))  # Position of bars for each algorithm

            mean_accuracies = []  # Store the mean accuracies for each algorithm
            std_devs = []  # Store the standard deviations for each algorithm
            for k, algorithm in enumerate(algorithms):
                # Load accuracy data for each algorithm-dataset-partitioner combination
                data = load_accuracy_data(algorithm, dataset, partitioner_type, partitioner_type, partitioner_file)

                if not data.empty:
                    # Extract the last 10 rounds (tail) and calculate the mean accuracy
                    tail_data = data.tail(30)
                    mean_accuracy = tail_data['Accuracy'].mean()
                    std_dev = tail_data['Accuracy'].std()  # Standard deviation of the last 10 rounds
                    mean_accuracies.append(mean_accuracy)
                    std_devs.append(std_dev)
                else:
                    mean_accuracies.append(0)  # Handle the case where there's no data
                    std_devs.append(0)  # No error if no data

            # Create a bar plot with error bars for the mean accuracies of all algorithms
            ax.bar(
                x_positions, mean_accuracies,
                width=bar_width,
                color=[algorithm_colors[alg] for alg in algorithms],
                #tick_label=algorithms,
                yerr=std_devs,  # Add error bars
                capsize=5  # Add caps to the error bars
            )

            # Set y-axis limits to [0, 1]
            ax.set_ylim([0, 1])

            # Set title for each subplot
            if partitioner_type == 'drichlet':
                title = f"Alpha {partitioner_file.replace('alpha', '')}"
            elif partitioner_type == 'pathological':
                title = f"Classes {partitioner_file.replace('classes', '')}"

            ax.set_title(f"{title}", fontsize=14)
            ax.grid(True)

            # Only set y-labels on the leftmost subplots
            if j == 0:
                ax.set_ylabel(f'{dataset}', fontsize=14)

    # Add common x and y labels for the entire figure
    fig.supxlabel('Algorithm', fontsize=18)
    fig.supylabel('Mean Accuracy (Tail 10 Rounds)', fontsize=18)

    # Create a single legend for all subplots
    handles = [plt.Rectangle((0, 0), 1, 1, color=algorithm_colors[alg]) for alg in algorithms]
    labels = algorithms
    fig.legend(handles, labels, loc='upper center', ncol=len(algorithms), fontsize=14, bbox_to_anchor=(0.5, 0.96))

    # Adjust layout to fit everything properly
    plt.tight_layout(rect=[0, 0, 1, 0.925])
    plt.savefig(os.path.join(base_dir, figure_title + '.pdf'))
    plt.show()

# Create two figures: one for Dirichlet partitioner and one for Pathological partitioner (Bar plot of Mean Tail 10 with error bars)
create_figure('Figure 3 poison: Dirichlet Partitioner Mean Accuracy Comparisons (Bar Plot Tail 10 with Error Bars)', 'drichlet')
create_figure('Figure 4 poison: Pathological Partitioner Mean Accuracy Comparisons (Bar Plot Tail 10 with Error Bars)', 'pathological')
