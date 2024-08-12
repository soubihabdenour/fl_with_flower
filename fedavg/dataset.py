# Download MNIST dataset and partition it
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from omegaconf import DictConfig


def get_data(partitions_number: int, config: DictConfig):

    fds = FederatedDataset(dataset="hf-vision/chest-xray-pneumonia", partitioners={"train": DirichletPartitioner(
            num_partitions=partitions_number,
            partition_by="label",
            alpha=1.0,
            min_partition_size=0,
        )})
    centralized_testset = fds.load_split("test")
    return fds, centralized_testset