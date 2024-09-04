from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from datasets import disable_progress_bar
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from hydra.utils import call, instantiate

from fedavg_mobilnet.client import get_client_fn
from fednova_mobilnet.dataset import get_data
from fedavg_mobilnet.server import fit_config, weighted_average, get_evaluate_fn
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
    fds, centralized_testset = get_data(partitions_number=cfg.num_clients, config=cfg.dataset, path=save_path)

    strategy = instantiate(
        cfg.strategy.strategy,
        evaluate_metrics_aggregation_fn=weighted_average,
        accept_failures=False,
        on_fit_config_fn=fit_config_fn,
        initial_parameters=init_parameters,
        evaluate_fn=eval_fn,
        fraction_evaluate=0.0,
        **extra_args,
    )
    histotory= fl.simulation.start_simulation(
        client_fn=get_client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=
    )

if __name__ == "__main__":
    main()