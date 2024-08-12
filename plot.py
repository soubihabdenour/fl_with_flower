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

def smooth_plot(history, title, smoothing_window=5):
    print(f"{history.metrics_centralized = }")

    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]

    # Apply smoothing
    if smoothing_window > 1:
        acc_smooth = moving_average(acc, smoothing_window)
        round_smooth = round[:len(acc_smooth)]
    else:
        acc_smooth = acc
        round_smooth = round

    plt.plot(round_smooth, acc_smooth, color="blue", label="Smoothed Accuracy")
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title(title)
    plt.legend()
    plt.show()
