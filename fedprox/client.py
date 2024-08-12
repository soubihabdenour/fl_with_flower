from collections import OrderedDict
from typing import Dict

import flwr as fl
import torch
from flwr.common import Scalar
from torch import nn
from torchvision import models

from fedprox.model import Net, train, test


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloder,
                 num_classes):
        super().__init__()

        self.trainloader = trainloader
        self.valloder = valloder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = Net(num_classes)

        #self.model = models.vgg16(pretrained=True).to(self.device)
        #self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)


    def set_parameters(self, parameters):
        parameters = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in parameters})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def fit(self, parameters, config):

        #copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        train(self.model, self.trainloader, optim, epochs=epochs)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloder, self.device)
        return float(loss), len(self.valloder), {'accuracy', accuracy}

def generate_client_fn(trainloaders, valloaders, num_classes):
    def client_fn(cid):

        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloder=valloaders[int(cid)],
                            num_classes=num_classes
                            ).to_client()

    return client_fn