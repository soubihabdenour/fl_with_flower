"""pytorchexample: A Flower / PyTorch app."""

from collections import OrderedDict
from typing import List
import flwr as fl
import torch
from flwr_datasets import FederatedDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from fedavg_vgg19.utils import train, test, apply_transforms
from torchvision.models import VGG16_Weights, VGG19_Weights



# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, num_classes):
        self.trainset = trainset
        self.valset = valset

        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        #self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        # Replace the last fully connected layer
        # VGG16's final classifier layer is at index 6
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_mal_parameters(self, config):
        return [val.cpu().numpy() * 1000 for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)

        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)

        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_client_fn(dataset: FederatedDataset, num_classes):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(context) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(
            int(context.node_config["partition-id"]), "train"
        )

        # Now let's split it into train (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)

        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FlowerClient(trainset, valset, num_classes).to_client()

    return client_fn
