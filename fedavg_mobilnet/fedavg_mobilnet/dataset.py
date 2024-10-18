from pathlib import Path
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner
from omegaconf import DictConfig
# from flwr_datasets.visualization import plot_label_distributions  # Uncomment if needed


def get_data(partitions_number: int, config: DictConfig, path: str):
    """
    Prepare the federated dataset based on the specified partitioning strategy in the configuration.

    Args:
        partitions_number (int): The number of dataset partitions (clients).
        config (DictConfig): Configuration object containing dataset and partitioner settings.
        path (str): Path to save visualizations of label distributions (optional).

    Returns:
        FederatedDataset: The federated dataset based on the partitioning strategy.
        Dataset: The centralized test set used for evaluation.
    """

    # Determine the partitioning strategy based on the configuration
    if config.partitioner.name == "DirichletPartitioner":
        # Non-IID partitioning using Dirichlet distribution
        print("Using Dirichlet partitioning (Non-IID)")
        fds = FederatedDataset(
            dataset=config.name,
            subset=config.subset,
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=partitions_number,
                    partition_by="label",  # Partition based on label distribution
                    seed=config.seed,
                    alpha=config.partitioner.alpha,  # Control label distribution skew
                    min_partition_size=0
                )
            }
        )
    elif config.partitioner.name == "PathologicalPartitioner":
        # Non-IID partitioning using pathological partitioning (e.g., limited classes per partition)
        print("Using Pathological partitioning (Non-IID)")
        fds = FederatedDataset(
            dataset=config.name,
            subset=config.subset,
            data_dir=config.data_dir,
            partitioners={
                "train": PathologicalPartitioner(
                    num_partitions=partitions_number,
                    seed=config.seed,
                    partition_by="label",
                    num_classes_per_partition=config.partitioner.num_classes_per_partition
                )
            }
        )
    elif config.partitioner.name == "IiD":
        # IID partitioning (evenly distributed samples across partitions)
        print("Using IID partitioning")
        fds = FederatedDataset(
            dataset=config.name,
            subset=config.subset,
            partitioners={"train": partitions_number}
        )
    else:
        raise ValueError(f"Unknown partitioner: {config.partitioner.name}")

    # Load centralized test set (used for global model evaluation)
    centralized_testset = fds.load_split("test")

    # Optional: Uncomment the following block if you want to visualize and save label distributions
    # fig2, ax, df = plot_label_distributions(
    #     fds.partitioners["train"],
    #     label_name="label",
    #     plot_type="bar",
    #     size_unit="absolute",
    #     partition_id_axis="x",
    #     legend=True,
    #     verbose_labels=True,
    #     title=None,
    #     legend_kwargs={"ncols": 1, "bbox_to_anchor": (0.9, 0.66)}
    # )
    # path = Path(path) / f"{config.subset}_{config.partitioner.name}.pdf"
    # fig2.savefig(path, bbox_inches='tight')

    return fds, centralized_testset