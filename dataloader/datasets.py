from flwr_datasets import FederatedDataset

from dataloader.partitioner import PartitioningFactory


class DataLoaderFactory:
    @staticmethod
    def create_dataset(config):
        dataset_name = config.dataset.name
        partitioner = PartitioningFactory.partitioner(config)
        if dataset_name == "Pneumonia":
            params = config.dataset.parameters
            return FederatedDataset(**params, partitioners={"train": partitioner})
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
