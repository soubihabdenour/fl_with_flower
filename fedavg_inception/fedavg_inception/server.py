from collections import OrderedDict
from typing import List, Tuple, Dict

import flwr as fl
import torch
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Metrics
from flwr.common.typing import Scalar
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from fedavg_inception.utils import test, apply_transforms
from torchvision.models import Inception_V3_Weights


def fit_config(epochs: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 2,  # Number of local epochs done by clients
        "batch_size": 64,  # Batch size to use by clients during fit()
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(centralized_testset: Dataset, num_classes: int):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the test set for evaluation."""

        # Load the InceptionV3 model
        model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)

        # Replace the fully connected (fc) layer with a new one for num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        # Move model to the appropriate device (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the model parameters from the federated learning server
        set_params(model, parameters)
        model.to(device)

        # Apply transformations to the centralized testset
        testset = centralized_testset.with_transform(apply_transforms)

        # Disable progress bar for dataset preprocessing (if applicable)
        disable_progress_bar()

        # Prepare the test loader
        testloader = DataLoader(testset, batch_size=32)

        # Run the test function and return the loss and accuracy
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate


def set_params(model: nn.Module, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
