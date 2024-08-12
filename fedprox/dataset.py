import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


def get_mnist(data_Dpath: str = "./data"):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_Dpath, train=True, download=True, transform=tr)
    testset = MNIST(data_Dpath, train=False, download=True, transform=tr)

    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, test_batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()

    #split partitions into 'numpartitions' trainsets
    num_images = len(trainset) // num_partitions
    partions_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partions_len, torch.Generator().manual_seed(2023))

    trainloaders = []
    valloaders = []
    #create dataloader with train+val support
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(num_total * val_ratio)
        num_train = num_total - num_val
        for_train, for_val = random_split(trainset_, [num_train, num_val])
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloader = DataLoader(testset, batch_size=test_batch_size)

    return trainloaders, valloaders, testloader