from collections import OrderedDict
from typing import List
import torch
from flwr_datasets import FederatedDataset
from torch import nn

from torch.utils.data import DataLoader
from tqdm import tqdm
import flwr as fl

from utils import Net, test, train, apply_transforms

from torchvision import models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # Instantiate model
        #self.model = Net()
        self.model = models.vgg16(pretrained=True).to(self.device)
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=2)

        # Determine device
        #self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    @staticmethod
    def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    def fit(self, parameters, config):
        self.set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_params(self.model, parameters)

        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)

        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
    @staticmethod
    def get_client_fn(dataset: FederatedDataset):
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
            return FlowerClient(trainset, valset).to_client()

        return client_fn
    @staticmethod
    def train(net, trainloader, epochs):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for batch in tqdm(trainloader, "Training"):
                images = batch["img"]
                labels = batch["label"]
                optimizer.zero_grad()
                criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
                optimizer.step()

    @staticmethod
    def test(net, testloader):
        """Validate the model on the test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in tqdm(testloader, "Testing"):
                images = batch["img"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        return loss, accuracy
