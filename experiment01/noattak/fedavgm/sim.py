import argparse
from flwr.client.mod.localdp_mod import LocalDpMod
import torch

import flwr as fl
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
import numpy as np
import plot
from clients import FlowerClient
from servers import server_fn, strategy
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)

NUM_CLIENTS = 100
NUM_ROUNDS = 10

# Download MNIST dataset and partition it
mnist_fds = FederatedDataset(dataset="MadElf1337/Pneumonia_Images", partitioners={"train": NUM_CLIENTS}, seed=seed)
centralized_testset = mnist_fds.load_split("test")

from flwr.server import ServerAppComponents

# Create an instance of the mod with the required params
local_dp_obj = LocalDpMod(
    0.1, 0.1, 0.1, 0.3
)
# ClientApp for Flower-Next
client = fl.client.ClientApp(
    client_fn=FlowerClient.get_client_fn(mnist_fds)
)

# ServerApp for Flower-Next
server = fl.server.ServerApp(server_fn=server_fn)


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Parse input arguments
    args = parser.parse_args()
    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=FlowerClient.get_client_fn(mnist_fds),
        num_clients=NUM_CLIENTS,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy("FedAvgM", proximal_mu=0.1),
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )
    plot.plot(history)


if __name__ == "__main__":
    main()
