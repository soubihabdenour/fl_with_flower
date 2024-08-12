from collections import OrderedDict

from omegaconf import DictConfig
from torch import nn
from torchvision import models
import torch

from fedprox.model import test


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):

        return {'lr':config.lr,
                'momentum':config.momentum,
                'local_epochs':config.local_epochs
                }

    return fit_config_fn

def get_evaluate_fn(num_classes: int, testloader):
    def evaluate_fn(server_round: int, parameters, config):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.vgg16(pretrained=True).to(device)
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)

        parameters = zip(model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in parameters})

        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)
        return loss, {'accuracy': accuracy}

    return evaluate_fn