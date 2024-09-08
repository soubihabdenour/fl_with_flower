import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
file_paths = [
    '/home/abdenour/PycharmProjects/fl_with_flower/fedavg_mobilnet/outputs/2024-08-30/12-15-19/PathologicalPartitioner_30.csv',
    '/home/abdenour/PycharmProjects/fl_with_flower/fedprox_mobilnet/outputs/2024-09-01/17-25-14/PathologicalPartitioner_30.csv',
    '/home/abdenour/PycharmProjects/fl_with_flower/fedavg_mobilnet/outputs/2024-08-30/12-15-19/PathologicalPartitioner_30.csv'
]

# Read the data from each file
data1 = pd.read_csv(file_paths[0])
data2 = pd.read_csv(file_paths[1])
data3 = pd.read_csv(file_paths[2])

# Extracting the columns for plotting
rounds = data1['Round'][::20]  # Selecting every 10th round
iid = data1['Accuracy'][::20]
dri = data2['Accuracy'][::20]
patho = data3['Accuracy'][::20]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(rounds, iid, color='blue', marker='^', linestyle='-', label='IiD')
plt.plot(rounds, dri, color='orange', marker='o', linestyle='-', label='Dirichlet')
plt.plot(rounds, patho, color='green', marker='s', linestyle='-', label='Pathological')

#plt.fill_between(rounds, iid - 0.1, iid + 0.1, alpha=0.3)
# Set x-axis to log scale


# Labels and title
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
plt.title('Comparison of FedAvg and Custom-FedAvgM Across Three Files')

# Legend
plt.legend()
plt.savefig('graph2.pdf', bbox_inches='tight')
# Display the plot
plt.show()
