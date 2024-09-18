import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.client.mod import LocalDpMod
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale, Resize


def comp_accuracy(output, target, topk=(1,)):
    """Compute accuracy over the k top predictions wrt the target."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
