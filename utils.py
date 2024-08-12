import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import ToTensor, Normalize, Compose, Resize, Grayscale, CenterCrop


# transformation to convert images to tensors and apply normalization
def apply_transforms(batch):
    transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv1.weight.data.normal_(0, 0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2.weight.data.normal_(0, 0.1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(120, 84)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(84, num_classes)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    #UNDER REVIEW
    def compute_size(self, size):
        x = torch.zeros(size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x= torch.flatten(x, 1)
        print(x.size())


# borrowed from Pytorch quickstart example
def train(net, trainloader, optim, epochs, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
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