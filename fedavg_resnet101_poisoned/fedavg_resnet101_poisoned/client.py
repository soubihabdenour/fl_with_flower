"""pytorchexample: A Flower / PyTorch app."""
import random
from collections import OrderedDict
from typing import List
import flwr as fl
import torch
from flwr_datasets import FederatedDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from fedavg_resnet101_poisoned.utils import train, test, apply_transforms




# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, num_classes, mal):
        self.trainset = trainset
        self.valset = valset
        self.mal = mal
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # Remove last layer and flatten outputs
        self.model = torch.nn.Sequential(
            *(list(self.model.children())[:-1]), torch.nn.Flatten(),
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        # Set the hidden_dimension
        self.model.hidden_dimension = 2048

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
        if self.mal == True:
            return self.get_mal_parameters({}), len(trainloader.dataset), {}
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


def get_client_fn(dataset: FederatedDataset, num_classes, poison_fraction ,mal_ids):
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
        # poison injection

        # Assuming partition is your dataset object
        num_to_poison = int(poison_fraction * len(client_dataset))  # Number of examples to poison
        #num_to_poison =len(client_dataset) # Number of examples to poison

        # Randomly choose indices to poison
        indices_to_poison = random.sample(range(len(client_dataset)), num_to_poison)

        possible_labels = list(range(num_classes))

        # Function to modify the 'label' column
        def update_label(example, idx):
            if idx in indices_to_poison:
                current_label = example['label']
                # Exclude the current label from possible choices and choose a new one randomly
                new_label = random.choice([label for label in possible_labels if label != current_label])
                #print('id', context.node_config["partition-id"],':', current_label,':', new_label,"-----------")
                example['label'] = new_label
            return example



        # Now let's split it into train (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)

        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]
        print(context.node_config["partition-id"], mal_ids, '=====================================================')
        if int(context.node_config["partition-id"]) in mal_ids:
            print('yes=======================================================================')
            # Apply the function to the dataset using `map`, and pass indices
            trainset = trainset.map(update_label, with_indices=True)
        else:
            print('no=======================================================================')
        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)
        # Create and return client
        return FlowerClient(trainset, valset, num_classes, mal=False).to_client()

    return client_fn
