import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(history, title):
    print(f"{history.metrics_centralized = }")

    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    #global_loss_centralised = history.metrics_centralized["loss"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    #loss = [data[1] for data in global_loss_centralised]
    plt.plot(round, acc, color="blue", label="Accuracy")
    #plt.plot(round, loss, color="red", label="loss")
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title(title)
    plt.legend()
    # Show plot
    plt.show()


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def smooth_plot(data, title, path, smoothing_window=5):
    print(f"{data.metrics_centralized = }")

    global_accuracy_centralised = data.metrics_centralized["accuracy"]
    round = [data[0] for data in global_accuracy_centralised]
    #acc = [100.0 * data[1] for data in global_accuracy_centralised]
    acc = [data[1] for data in global_accuracy_centralised]

    # Apply smoothing
    if smoothing_window > 1:
        acc_smooth = moving_average(acc, smoothing_window)
        round_smooth = round[:len(acc_smooth)]
    else:
        acc_smooth = acc
        round_smooth = round

    # Save the smoothed results to a CSV file
    csv_path = Path(path) / 'smoothed_results.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Accuracy"])
        writer.writerows(zip(round_smooth, acc_smooth))

    print(f"Results saved to {csv_path}")
    plt.plot(round_smooth, acc_smooth, color="blue", label="Accuracy")
    plt.grid()
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    #plt.title(title)
    plt.legend()
    plt.savefig(path / 'graph.png', bbox_inches='tight', )
    plt.savefig(path / 'graph.pdf', bbox_inches='tight')
    #plt.show()
    print("==========================================================", path)

