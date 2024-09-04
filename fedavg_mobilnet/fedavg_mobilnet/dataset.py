from flwr_datasets import FederatedDataset
from pathlib import Path
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner
from fontTools.subset import subset
from omegaconf import DictConfig
from flwr_datasets.visualization import plot_label_distributions


def get_data(partitions_number: int, config: DictConfig, path):
    if config.partitioner.name == "DirichletPartitioner":  # Non IiD
        print("drichlet_________________")
        fds = FederatedDataset(dataset=config.name,
                               subset=config.subset,
                               partitioners={"train": DirichletPartitioner(
                                   num_partitions=partitions_number,
                                   partition_by="label",
                                   seed=config.seed,
                                   alpha=config.partitioner.alpha,
                                   min_partition_size=0,
                               )})
    elif config.partitioner.name == "PathologicalPartitioner":  # Non Iid
        print("pathological__________________")
        fds = FederatedDataset(dataset=config.name,
                               subset=config.subset,
                               data_dir=config.data_dir,
                               partitioners={"train": PathologicalPartitioner(
                                   num_partitions=partitions_number,
                                   seed=config.seed,
                                   partition_by="label",
                                   num_classes_per_partition=config.partitioner.num_classes_per_partition,
                               )})
    elif config.partitioner.name == "IiD":  # IiD
        print("iid______________")
        fds = FederatedDataset(dataset=config.name,
                               subset=config.subset,
                               partitioners={"train": partitions_number})

    centralized_testset = fds.load_split("test")

    # fig2, ax, df = plot_label_distributions(
    #     fds.partitioners["train"],
    #     label_name="label",
    #     plot_type="bar",
    #     size_unit="absolute",
    #     partition_id_axis="x",
    #     legend=True,
    #     verbose_labels=True,
    #     title=None,
    #     legend_kwargs={"ncols": 1, "bbox_to_anchor": (0.9, 0.66), }
    # )
    # path = Path(path) / f"{config.subset}_{config.partitioner.name}.pdf"
    # fig2.savefig(path, bbox_inches='tight')
    return fds, centralized_testset
