import torch
import torch.nn as nn
from .builder import Builder
import torch.nn.functional as F

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
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

class Net2(nn.Module):
    def __init__(self, num_classes=2):
        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(256 * 14 * 14, 1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.linear2 = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.linear2(x)
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