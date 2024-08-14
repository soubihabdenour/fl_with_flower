from datasets.utils.logging import disable_progress_bar
from flwr.common.typing import Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch.nn as nn
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Parameters
from fedavg.utils import test, apply_transforms

from collections import OrderedDict
from typing import List, Tuple, Dict
import flwr as fl
from datasets import Dataset
from flwr.common import Metrics

from pyarrow import Scalar
import torch

from fedbn.utils import NetWithBnAndFrozen


def fit_config(config: DictConfig):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": config.local_epochs,  # Number of local epochs done by clients
        "batch_size": config.client_batch,  # Batch size to use by clients during fit()
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

def metric_aggregation(
    all_client_metrics: List[Tuple[int, Metrics]],
) -> Tuple[int, Metrics]:
    """
    Function that computes a weighted aggregation of metrics normalized by the total number of samples.

    Args:
        all_client_metrics (List[Tuple[int, Metrics]]): A list of tuples with the
            sample counts and metrics for each client.

    Returns:
        Tuple[int, Metrics]: The total number of examples along with aggregated metrics.
    """
    aggregated_metrics: Metrics = {}
    total_examples = 0
    # Run through all of the metrics
    for num_examples_on_client, client_metrics in all_client_metrics:
        total_examples += num_examples_on_client
        for metric_name, metric_value in client_metrics.items():
            # Here we assume each metric is normalized by the number of examples on the client. So we scale up to
            # get the "raw" value
            if isinstance(metric_value, float):
                current_metric_value = aggregated_metrics.get(metric_name, 0.0)
                assert isinstance(current_metric_value, float)
                aggregated_metrics[metric_name] = current_metric_value + num_examples_on_client * metric_value
            elif isinstance(metric_value, int):
                current_metric_value = aggregated_metrics.get(metric_name, 0)
                assert isinstance(current_metric_value, int)
                aggregated_metrics[metric_name] = current_metric_value + num_examples_on_client * metric_value
            else:
                raise ValueError("Metric type is not supported")
    return total_examples, aggregated_metrics

def normalize_metrics(total_examples: int, aggregated_metrics: Metrics) -> Metrics:
    """
    Function that normalizes metrics by provided sample count.

    Args:
        total_examples (int): The total number of samples across all client datasets.
        aggregated_metrics (Metrics): Metrics that have been aggregated across clients.

    Returns:
        Metrics: The metrics normalized by total_examples.
    """
    # Normalize all metric values by the total count of examples seen.
    normalized_metrics: Metrics = {}
    for metric_name, metric_value in aggregated_metrics.items():
        if isinstance(metric_value, float) or isinstance(metric_value, int):
            normalized_metrics[metric_name] = metric_value / total_examples
    return normalized_metrics
def evaluate_metrics_aggregation_fn(
    all_client_metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Function for evaluate that computes a weighted aggregation of the client metrics
    and normalizes by the total number of samples.

    Args:
        all_client_metrics (List[Tuple[int, Metrics]]): A list of tuples with the
            sample counts and metrics for each client.

    Returns:
        Metrics: The aggregated normalized metrics.
    """
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def get_all_model_parameters(model: nn.Module) -> Parameters:
    """
    Function to extract ALL parameters associated with a pytorch module, including any state parameters. These
    values are converted from numpy arrays into a Flower Parameters object.

    Args:
        model (nn.Module): PyTorch model whose parameters are to be extracted

    Returns:
        Parameters: Flower Parameters object containing all of the target models state.
    """
    # Extracting all model parameters and converting to Parameters object
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])
def get_evaluate_fn(centralized_testset: Dataset, num_classes: int):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
        """Use the entire CIFAR-10 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = NetWithBnAndFrozen(num_classes=num_classes, freeze_cnn_layer=False)


        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
