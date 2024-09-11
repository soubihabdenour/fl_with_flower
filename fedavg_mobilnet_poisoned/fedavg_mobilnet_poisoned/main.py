import random
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from datasets import disable_progress_bar
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from fedavg_mobilnet_poisoned.client import get_client_fn
from fedavg_mobilnet_poisoned.dataset import get_data
from fedavg_mobilnet_poisoned.server import fit_config, weighted_average, get_evaluate_fn
from torch.cuda import seed_all

from plot import smooth_plot


@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = Path(HydraConfig.get().runtime.output_dir)
    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus,
    }
    random.seed(0)
    #mal_ids = random.sample(range(0, cfg.num_clients), int(cfg.fraction_mal_clients * cfg.num_clients))
    mal_ids = random.sample(range(0, cfg.num_clients), int(1.0 * cfg.num_clients))
    print(mal_ids, '====================================================')
    fds, centralized_testset = get_data(partitions_number=cfg.num_clients, config=cfg.dataset, path=save_path)
    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(fds, num_classes=cfg.model.num_classes, poison_fraction=cfg.poison_fraction, mal_ids=mal_ids),
        num_clients=cfg.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=cfg.fraction_train_clients,  # Sample 10% of available clients for training
            fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
            min_available_clients=3,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
            evaluate_fn=get_evaluate_fn(centralized_testset, cfg.model.num_classes),  # Global evaluation function

        ),
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )

    smooth_plot(data=history,
                title=f"{cfg.dataset.name.split('/')[-1]} - {cfg.dataset.partitioner.name.split('Partitioner')[0]} - {cfg.num_clients} clients with 10 per round",
                path=save_path,
                smoothing_window=cfg.plot.smoothing_window)


if __name__ == "__main__":
    main()
