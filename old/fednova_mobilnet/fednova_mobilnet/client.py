"""Client implementation for FedNova."""

from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from flwr_datasets import FederatedDataset
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from flwr.common import Context
from fednova_mobilnet.utils import test, train, apply_transforms
from torch import nn
from torchvision import models

class FedNovaClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_classes,
        #client_id: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        ratio: float,
        config: DictConfig,
    ):
        #self.net = net
        self.net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.exp_config = config

        if self.exp_config.var_local_epochs and (
            self.exp_config.exp_name == "proximal"
        ):
            # For only FedNova with proximal local solver and variable local epochs,
            # mu = 0.001 works best.
            # For other experiments, the default setting of mu = 0.005 works best
            # Ref: https://arxiv.org/pdf/2007.07481.pdf (Page 33, Section:
            # More Experiment Details)
            self.exp_config.optimizer.mu = 0.001

        self.optimizer = instantiate(
            self.exp_config.optimizer, params=self.net.parameters(), ratio=ratio
        )
        self.trainloader = trainloader
        self.valloader = valloader
        #self.client_id = client_id
        # self.device = device
        # self.num_epochs = num_epochs
        self.data_ratio = ratio

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        params = [
            val["cum_grad"].cpu().numpy()
            for _, val in self.optimizer.state_dict()["state"].items()
        ]
        return params

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        self.optimizer.set_model_params(parameters)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        self.optimizer.set_lr(config["lr"])

        # if self.exp_config.var_local_epochs:
        #     seed_val = (
        #         2023
        #         + int(self.client_id)
        #         + int(config["server_round"])
        #         + int(self.exp_config.seed)
        #     )
        #     np.random.seed(seed_val)
        #     num_epochs = np.random.randint(
        #         self.exp_config.var_min_epochs, self.exp_config.var_max_epochs
        #     )
        # else:
        #     num_epochs = self.num_epochs

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]
        train(
            self.net, self.optimizer, self.trainloader, self.device, epochs=epochs
        )

        # Get ratio by which the strategy would scale local gradients from each client
        # We use this scaling factor to aggregate the gradients on the server
        grad_scaling_factor: Dict[str, float] = self.optimizer.get_gradient_scaling()

        return self.get_parameters({}), len(self.trainloader), grad_scaling_factor

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        # Evaluation ideally is done on validation set, but because we already know
        # the best hyper-parameters from the paper and since individual client
        # datasets are already quite small, we merge the validation set with the
        # training set and evaluate on the training set with the aggregated global
        # model parameters. This behaviour can be modified by passing the validation
        # set in the below test(self.valloader) function and replacing len(
        # self.valloader) below. Note that we evaluate on the centralized test-set on
        # server-side in the strategy.

        self.set_parameters(parameters)
        loss, metrics = test(self.net, self.trainloader, self.device)
        return float(loss), len(self.trainloader), metrics


def get_client_fn(  # pylint: disable=too-many-arguments
    dataset: FederatedDataset,
    num_classes,
    data_sizes: List,
    exp_config: DictConfig,
) -> Callable[[str], FedNovaClient]:
    """Return a generator function to create a FedNova client."""
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


        return FedNovaClient(
            num_classes,
            trainset,
            valset,
            ratio=0.5.
            #client_dataset_ratio,
            exp_config,
        )


    return client_fn
