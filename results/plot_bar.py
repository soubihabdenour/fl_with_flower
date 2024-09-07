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

# Function to load accuracy data from a CSV file
def load_accuracy_data(algorithm, dataset, partitioner, partitioner_type, file_name):
    csv_path = os.path.join(base_dir, algorithm, dataset, partitioner, file_name + '.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"File {csv_path} does not exist.")
        return pd.DataFrame()

# Function to create 3x3 subplots for given algorithms, datasets, and partitioners (Bar Plot for Tail 10 Rounds)
def create_figure(figure_title, partitioner_type):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(figure_title, fontsize=16)

    for i, dataset in enumerate(datasets):  # Row for each dataset
        partitioner_files = partitioners[partitioner_type]  # Get the files for the current partitioner
        for j, partitioner_file in enumerate(partitioner_files):  # Column for each partitioner
            ax = axs[i, j]
            bar_width = 0.15  # Width of each bar
            rounds = None  # Variable to store the round numbers
            for k, algorithm in enumerate(algorithms):
                # Load accuracy data for each algorithm-dataset-partitioner combination
                data = load_accuracy_data(algorithm, dataset, partitioner_type, partitioner_type, partitioner_file)

                if not data.empty:
                    # Extract the last 10 rounds (tail)
                    tail_data = data.tail(10)
                    rounds = tail_data['Round'] if rounds is None else rounds  # Set rounds if not already set

                    # Compute positions for the bars for each algorithm
                    x_positions = np.arange(len(rounds)) + k * bar_width
                    ax.bar(x_positions, tail_data['Accuracy'], width=bar_width, label=algorithm)

            # Set labels and titles for each subplot
            if partitioner_type == 'drichlet':
                title = f"Drichlet - Alpha {partitioner_file.replace('alpha', '').replace('csv', '')}"
            elif partitioner_type == 'pathological':
                title = f"Pathological - Classes {partitioner_file.replace('classes', '')}"

            ax.set_title(f"{title} - {dataset}")
            ax.set_xlabel('Round')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(np.arange(len(rounds)) + bar_width * len(algorithms) / 2)
            ax.set_xticklabels(rounds.astype(int))  # Ensure rounds are integers for display
            ax.legend(loc='best')
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(base_dir, figure_title + '.pdf'))
    plt.show()


# Create two figures: one for Dirichlet partitioner and one for Pathological partitioner (Bar plot of Tail 10)
create_figure('Figure 1: Dirichlet Partitioner Accuracy Comparisons (Bar Plot Tail 10)', 'drichlet')
create_figure('Figure 2: Pathological Partitioner Accuracy Comparisons (Bar Plot Tail 10)', 'pathological')
