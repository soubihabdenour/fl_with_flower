import torch
import torch.nn as nn
from .builder import Builder
import torch.nn.functional as F

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNetGry(nn.Module):
    def __init__(self):
        super(LeNetGry, self).__init__()
        builder = Builder()
        self.convs = nn.Sequential(
            builder.conv3x3(1, 32, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(32, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.linear = nn.Sequential(
            builder.conv1x1(1048576, 128),
            nn.ReLU(),
            builder.conv1x1(128, 10),
        )

    def forward(self, x):
        out = x.view(x.size(0), 1, 256, 256)
        out = self.convs(out)
        out = out.view(out.size(0), 64 * 128 * 128, 1, 1)
        out = self.linear(out)
        return out.squeeze()