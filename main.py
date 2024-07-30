import hydra
from omegaconf import DictConfig, OmegaConf

from dataloader.datasets import DataLoaderFactory


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    config = cfg
    dataset = DataLoaderFactory.create_dataset(config)
    data = dataset.load_partition(0)


if __name__ == "__main__":
    main()