from flwr_datasets.partitioner import IidPartitioner, Partitioner


class CustomPartioner(Partitioner):
    pass


class PartitioningFactory:
    @staticmethod
    def partitioner(config):
        strategy_name = config.partitioning.name
        params = config.partitioning.parameters
        num_partition = config.clients.client_number
        if strategy_name == "iid":
            return IidPartitioner(num_partitions=num_partition)
        # elif strategy_name == "non_iid":
        #    return NonIIDPartitioning()
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy_name}")
