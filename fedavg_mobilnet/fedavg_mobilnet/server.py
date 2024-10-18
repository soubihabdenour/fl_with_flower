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

from fedavg_mobilnet.utils import test, apply_transforms


def fit_config(epochs: int) -> Dict[str, Scalar]:
    """
    Return a configuration with static batch size and local epochs for each client.

    Args:
        epochs (int): Number of local epochs to be run by each client.

    Returns:
        Dict[str, Scalar]: A dictionary containing the configuration for the client.
    """
    config = {
        "epochs": epochs,  # Number of local epochs performed by each client
        "batch_size": 64,  # Static batch size used by clients during training
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregation function to compute the weighted average of evaluation metrics (e.g., accuracy)
    across multiple clients.

    Args:
        metrics (List[Tuple[int, Metrics]]): List of tuples containing the number of examples
        and the metrics (accuracy) from each client.

    Returns:
        Metrics: Aggregated metrics using a weighted average based on the number of examples.
    """
    # Calculate weighted accuracies
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Return aggregated accuracy (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(centralized_testset: Dataset, num_classes: int):
    """
    Return an evaluation function to perform centralized evaluation using the test set.

    Args:
        centralized_testset (Dataset): The centralized test dataset.
        num_classes (int): The number of classes in the classification task.

    Returns:
        Callable: A function that performs evaluation given the server round, model parameters,
        and evaluation config.
    """

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
        """
        Evaluate the global model on the centralized test set.

        Args:
            server_round (int): The current round of federated learning.
            parameters (fl.common.NDArrays): Model parameters received from the server.
            config (Dict[str, Scalar]): Evaluation configuration.

        Returns:
            Tuple[float, Metrics]: Loss value and a dictionary containing evaluation metrics.
        """

        # Load pre-trained MobileNetV2 model and adjust the classifier layer for the specific task
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        # Set device to GPU if available, else fallback to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model parameters
        set_params(model, parameters)
        model.to(device)

        # Apply the necessary transformations to the test set
        testset = centralized_testset.with_transform(apply_transforms)

        # Disable progress bars for the test set
        disable_progress_bar()

        # Create a DataLoader for the test set
        testloader = DataLoader(testset, batch_size=32)

        # Evaluate the model on the test set and return loss and accuracy
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate


def set_params(model: nn.Module, params: List[fl.common.NDArrays]):
    """
    Set the weights of the model using the provided list of NumPy arrays (parameters).

    Args:
        model (nn.Module): The PyTorch model whose parameters need to be set.
        params (List[fl.common.NDArrays]): A list of NumPy arrays representing the model parameters.
    """
    # Convert the list of NumPy arrays to a state dict and load it into the model
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)