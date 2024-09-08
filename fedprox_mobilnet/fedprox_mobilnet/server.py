from omegaconf import DictConfig
from torchvision import models
from torch import nn
import torch
from datasets import Dataset
import flwr as fl
from typing import List, Tuple, Dict
from flwr.common.typing import Scalar
from datasets.utils.logging import disable_progress_bar
from collections import OrderedDict
from torch.utils.data import DataLoader
from fedprox_mobilnet.utils import test, apply_transforms

def get_on_fit_config(config: DictConfig):
    """Generate the function for config.

    The config dict is sent to the client fit() method.
    """

    def fit_config_fn(server_round: int):  # pylint: disable=unused-argument
        # option to use scheduling of learning rate based on round
        # if server_round > 50:
        #     lr = config.lr / 10
        return {
            "local_epochs": config.local_epochs,
            "batch_size": config.local_batch_size,
        }

    return fit_config_fn

def get_evaluate_fn(centralized_testset: Dataset, num_classes: int):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the test set for evaluation."""

        # Determine device
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = models.MobileNetV2(num_classes=2)
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=2)

        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=32)
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)