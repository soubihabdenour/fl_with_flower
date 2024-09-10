import pandas as pd
import matplotlib.pyplot as plt

# File paths
file_paths = [
    '/home/abdenour/PycharmProjects/fl_with_flower/fedavg_mobilnet_poisoned/outputs/2024-08-30/12-15-19/PathologicalPartitioner_30.csv',
    '/home/abdenour/PycharmProjects/fl_with_flower/fedprox_mobilnet/outputs/2024-09-01/17-25-14/PathologicalPartitioner_30.csv',
    '/home/abdenour/PycharmProjects/fl_with_flower/fedavg_mobilnet_poisoned/outputs/2024-08-30/12-15-19/PathologicalPartitioner_30.csv'
]

# Load data
dataframes = [pd.read_csv(file) for file in file_paths]

# Extract the last 10 accuracies for each dataframe
last_10_accuracies = [df['Accuracy'].tail(10) for df in dataframes]

# Calculate the mean and standard deviation
means = [acc.mean() for acc in last_10_accuracies]
std_devs = [acc.std() for acc in last_10_accuracies]

# Plot error bars
plt.figure(figsize=(10, 6))
plt.bar(range(1, 4), means, yerr=std_devs, capsize=5, color='blue')

# Set plot labels and title
plt.xlabel('Data distribution')
plt.ylabel('Accuracy')
plt.xticks([1, 2, 3], ['Iid', 'Dirichlet', 'Pathological'])
plt.title('Error Bar Plot of the Last 10 Accuracies from Each File')
plt.savefig('error_bar_plot.pdf', bbox_inches='tight')
plt.show()
