import matplotlib.pyplot as plt

def plot(history):
    print(f"{history.metrics_centralized = }")

    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    plt.plot(round, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("MNIST - IID - 100 clients with 10 clients per round")
    plt.legend()
    # Show plot
    plt.show()