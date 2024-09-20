"""pytorchexample: A Flower / PyTorch app."""

from collections import OrderedDict
from typing import List
import flwr as fl
import numpy as np
import torch
from flwr_datasets import FederatedDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from typing import Callable, Dict, List, Tuple
from torchvision.models import VGG16_Weights, VGG19_Weights

from fednova_vgg16.utils import train, test, apply_transforms
from fednova_vgg16.models import ProxSGD


# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, num_classes, exp_config, ratio, cid):
        self.trainset = trainset
        self.valset = valset
        self.data_ratio = ratio
        self.exp_config= exp_config
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        # self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        # Replace the last fully connected layer
        # VGG16's final classifier layer is at index 6
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
        self.client_id = cid["partition-id"]
        print(cid,"cid================================")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = ProxSGD(self.model.parameters(),lr= 0.01, momentum=0, weight_decay=1e-4, mu=0.005, ratio=self.data_ratio)
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
        if self.exp_config.var_local_epochs:
            print('pot==================================')
            seed_val = (
                    2023
                    + int(self.client_id)
                    + int(self.exp_config.seed)
            )
            np.random.seed(seed_val)
            num_epochs = np.random.randint(
                self.exp_config.var_min_epochs, self.exp_config.var_max_epochs
            )
        else:
            num_epochs = self.num_epochs
        # Train
        train(self.model, trainloader, self.optimizer, epochs=num_epochs, device=self.device)

        # Get ratio by which the strategy would scale local gradients from each client
        # We use this scaling factor to aggregate the gradients on the server
        grad_scaling_factor: Dict[str, float] = self.optimizer.get_gradient_scaling()

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), grad_scaling_factor

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


def get_client_fn(dataset: FederatedDataset, num_classes, config_fit):
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
        client_datasize= len(client_dataset)

        client_dataset_ratio= float(client_datasize / dataset.partitioners['train'].dataset.dataset_size)

        print(client_dataset_ratio,'size=======================', dataset.partitioners['train'].dataset.dataset_size)

        # Now let's split it into train (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)

        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FlowerClient(trainset, valset, num_classes, config_fit, client_dataset_ratio, context.node_config).to_client()

    return client_fn
