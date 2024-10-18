"""
pytorchexample: A Flower / PyTorch app using Federated Learning.

This example uses the Flower federated learning framework (flwr) and PyTorch's 
MobileNetV2 architecture to perform model training and evaluation across multiple clients.

Clients load their own partitions of the dataset, apply transformations, 
and participate in federated learning by sharing model updates with the server.
"""

from collections import OrderedDict
from typing import List
import flwr as fl
import torch
from flwr_datasets import FederatedDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from fedavg_mobilnet.utils import train, test, apply_transforms


# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    """
    FlowerClient is responsible for training and evaluating the model locally on
    the client dataset and sharing the updated model parameters with the server.
    """

    def __init__(self, trainset, valset, num_classes: int):
        """
        Initialize the FlowerClient with the dataset and model.

        Args:
            trainset: The training dataset.
            valset: The validation dataset.
            num_classes: Number of output classes for the model.
        """
        self.trainset = trainset
        self.valset = valset

        # Load the pre-trained MobileNetV2 model and modify the classifier for the task.
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

        # Set device to GPU if available, otherwise fallback to CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def get_parameters(self, config):
        """
        Get model parameters in the form of a list of NumPy arrays.

        Args:
            config: Configuration for fetching parameters (not used here).

        Returns:
            A list of NumPy arrays representing the model's current parameters.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_mal_parameters(self, config):
        """
        (Example function) Get manipulated parameters by multiplying by 1000.

        Args:
            config: Configuration for fetching parameters (not used here).

        Returns:
            A list of manipulated NumPy arrays representing the model's parameters.
        """
        return [val.cpu().numpy() * 1000 for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """
        Train the model on the local dataset.

        Args:
            parameters: Model parameters received from the server.
            config: A dictionary with 'batch_size' and 'epochs' as keys.

        Returns:
            Updated model parameters, the number of training examples, and any other
            optional information (empty dict here).
        """
        # Set the received parameters in the local model
        set_params(self.model, parameters)

        # Read batch size and number of epochs from config
        batch_size, epochs = config["batch_size"], config["epochs"]

        # Construct dataloader for training
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        # Define optimizer (Adam with learning rate 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model locally
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        # Return updated parameters, dataset size, and an empty dictionary
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the local validation dataset.

        Args:
            parameters: Model parameters received from the server.
            config: Configuration for evaluation (not used here).

        Returns:
            Loss value, size of the validation dataset, and accuracy.
        """
        # Set the received parameters in the local model
        set_params(self.model, parameters)

        # Construct dataloader for validation
        valloader = DataLoader(self.valset, batch_size=64)

        # Evaluate the model on validation data
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Return loss, dataset size, and accuracy
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def set_params(model: torch.nn.Module, params: List[fl.common.NDArrays]):
    """
    Set the model's parameters from a list of NumPy arrays.

    Args:
        model: The PyTorch model.
        params: A list of NumPy arrays representing the parameters.
    """
    # Create a state dict by mapping each parameter to its corresponding key
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    # Load the new state dict into the model
    model.load_state_dict(state_dict, strict=True)


def get_client_fn(dataset: FederatedDataset, num_classes: int):
    """
    Return a function to construct a Flower client.

    Args:
        dataset: The federated dataset containing multiple client partitions.
        num_classes: Number of output classes for the model.

    Returns:
        A function that creates a Flower client for the specific dataset partition.
    """

    def client_fn(context) -> fl.client.Client:
        """
        Construct a FlowerClient with its own dataset partition.

        Args:
            context: Context with client-specific configuration.

        Returns:
            A FlowerClient instance for the assigned dataset partition.
        """
        # Get the partition ID for the client from the context
        partition_id = int(context.node_config["partition-id"])

        # Load the corresponding dataset partition
        client_dataset = dataset.load_partition(partition_id, "train")

        # Split the dataset into training (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)
        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        # Apply transformations to the training and validation datasets
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return the FlowerClient
        return FlowerClient(trainset, valset, num_classes).to_client()

    return client_fn