import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.client.mod import LocalDpMod
from omegaconf import DictConfig
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale, Resize
from torchvision import models


def fit_config(exp_config: DictConfig, server_round: int):
    """Return training configuration dict for each round.

    Learning rate is reduced by a factor after set rounds.
    """
    config = {}

    lr = exp_config.config_fit.lr

    if exp_config.lr_scheduling:
        if server_round == int(exp_config.num_rounds / 2):
            lr = exp_config.config_fit.lr / 10

        elif server_round == int(exp_config.num_rounds * 0.75):
            lr = exp_config.config_fit.lr / 100

    config["lr"] = lr
    config["server_round"] = server_round
    return config
# transformation to convert images to tensors and apply normalization
def apply_transforms(batch):
    tf = Compose([
        Resize((224, 224)),
        Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply transformations
    batch["image"] = [tf(img) for img in batch["image"]]
    return batch


def get_local_dp(config):
    return LocalDpMod(
        config.clipping_norm,
        config.sensitivity,
        config.epsilon,
        config.delta
    )


# borrowed from Pytorch quickstart example
def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()


# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
