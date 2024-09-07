import os
import pandas as pd
import matplotlib.pyplot as plt

# Define your datasets, partitioners, and algorithms
datasets = ['BloodMNIST', 'PathMNIST', 'TissueMNIST']
algorithms = ['fedavg', 'fedavgm', 'fedprox', 'fedbn', 'fednova']
partitioners = {
    'drichlet': ['alpha0.9', 'alpha0.3', 'alpha0.1'],
    'pathological': ['classes07', 'classes04', 'classes02']
}

# Base directory where the CSV results are stored
base_dir = '/home/abdenour/PycharmProjects/fl_with_flower/results'


# Function to load accuracy data from a CSV file
def load_accuracy_data(algorithm, dataset, partitioner, partitioner_type, file_name):
    if partitioner_type == 'drichlet':
        csv_path = os.path.join(base_dir, algorithm, dataset, partitioner,
                                file_name + '.csv')  # file_name already includes 'csv'
    elif partitioner_type == 'pathological':
        csv_path = os.path.join(base_dir, algorithm, dataset, partitioner, file_name + '.csv')  # append '.csv'

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"File {csv_path} does not exist.")
        return pd.DataFrame()


# Function to create 3x3 subplots for given algorithms, datasets, and partitioners
def create_figure(figure_title, partitioner_type):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(figure_title, fontsize=16)

    for i, dataset in enumerate(datasets):  # Row for each dataset
        partitioner_files = partitioners[partitioner_type]  # Get the files for the current partitioner
        for j, partitioner_file in enumerate(partitioner_files):  # Column for each partitioner
            ax = axs[i, j]
            for algorithm in algorithms:
                # Load accuracy data for each algorithm-dataset-partitioner combination
                data = load_accuracy_data(algorithm, dataset, partitioner_type, partitioner_type, partitioner_file)

                if not data.empty:
                    ax.plot(data['Round'], data['Accuracy'], label=algorithm)

            # Set labels and titles for each subplot
            if partitioner_type == 'drichlet':
                title = f"Drichlet - Alpha {partitioner_file.replace('alpha', '').replace('csv', '')}"
            elif partitioner_type == 'pathological':
                title = f"Pathological - Classes {partitioner_file.replace('classes', '')}"

            ax.set_title(f"{title} - {dataset}")
            ax.set_xlabel('Round')
            ax.set_ylabel('Accuracy')
            ax.legend(loc='best')
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(base_dir, figure_title + '.pdf'))
    plt.show()


# Create two figures: one for Dirichlet partitioner and one for Pathological partitioner
create_figure('Figure 1: Dirichlet Partitioner Accuracy Comparisons', 'drichlet')
create_figure('Figure 2: Pathological Partitioner Accuracy Comparisons', 'pathological')