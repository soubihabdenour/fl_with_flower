from flwr_datasets import FederatedDataset
NUM_CLIENTS = 100
# Download MNIST dataset and partition it
mnist_fds = FederatedDataset(dataset="Mnist", partitioners={"train": NUM_CLIENTS})
centralized_testset = mnist_fds.load_split("test")
class PneumoniaDataset():
    """
    dataset_name = "pneumonia"
    num_partitions = NUM_CLIENTS
    """
    def __init__(self, dataset="mnist", partitioners={"train": NUM_CLIENTS}):
        self.fds = FederatedDataset(dataset=dataset, partitioners=partitioners)
       # self.centralized_testset = self.load_split("test")