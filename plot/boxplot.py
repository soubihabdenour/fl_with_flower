import pandas as pd
import matplotlib.pyplot as plt
files = ['/home/abdenour/PycharmProjects/fl_with_flower/fedavg_mobilnet/outputs/2024-08-28/11-33-19/PathologicalPartitioner_30.csv',
         '/home/abdenour/PycharmProjects/fl_with_flower/fedavg_mobilnet/outputs/2024-08-28/11-15-55/PathologicalPartitioner_30.csv',
         '/home/abdenour/PycharmProjects/fl_with_flower/fedavg_mobilnet/outputs/2024-08-28/11-54-01/PathologicalPartitioner_30.csv']
dataframes = [pd.read_csv(file) for file in files]
last_10_accuracies = [df['Accuracy'].tail(8) for df in dataframes]
plt.figure(figsize=(10, 6))
plt.boxplot(last_10_accuracies, patch_artist=True)
#plt.title('Box Plot of the Last 10 Accuracies from Each File')
plt.xlabel('Data distribution')
plt.ylabel('Accuracy')
plt.xticks([1, 2, 3], ['Iid', 'Dirichlet', 'Pathological'])
plt.savefig('graph.pdf', bbox_inches='tight')
plt.show()
