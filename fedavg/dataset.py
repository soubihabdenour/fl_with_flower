from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from omegaconf import DictConfig


def get_data(partitions_number: int, config: DictConfig):

    if config.partitioner == "DirichletPartitioner":  # Non IiD
        fds = FederatedDataset(dataset=config.name, partitioners={"train": DirichletPartitioner(
            num_partitions=partitions_number,
            partition_by="label",
            alpha=config.partitioner.alpha,
            min_partition_size=0,
        )})
    elif config.partitioner == "PathologicalPartitioner":  # Non Iid
        fds = FederatedDataset(dataset=config.name, partitioners={"train": DirichletPartitioner(
            num_partitions=partitions_number,
            partition_by="label",
            num_classes_per_partition=config.partitioner.num_classes_per_partition,
            min_partition_size=0,
        )})
    else:  # IiD
        fds = FederatedDataset(dataset=config.name, partitioners={"train": partitions_number})

    centralized_testset = fds.load_split("test")

    return fds, centralized_testset