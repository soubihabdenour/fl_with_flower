from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from datasets import disable_progress_bar
from omegaconf import DictConfig, OmegaConf
import flwr as fl
import torch
from hydra.utils import call, instantiate
from functools import partial
from fednova_mobilnet.client import get_client_fn
from fednova_mobilnet.dataset import get_data
from fednova_mobilnet.server import  get_evaluate_fn, weighted_average
from flwr.common import ndarrays_to_parameters
from fednova_mobilnet.utils import fit_config, test


from fednova_mobilnet.strategy import FedNova
from plot import smooth_plot


@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(HydraConfig.get().runtime.output_dir)
    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus,
    }
    fds, centralized_testset = get_data(partitions_number=cfg.num_clients, config=cfg.dataset, path=save_path)
    # ndarrays = [
    #     layer_param.cpu().numpy()
    #     for _, layer_param in instantiate(cfg.model).state_dict().items()
    # ]
    # init_parameters = ndarrays_to_parameters(ndarrays)
    fit_config_fn = partial(fit_config, cfg)
    eval_fn = partial(test, instantiate(cfg.model), centralized_testset, device)
    strategy = instantiate(
        FedNova,
        evaluate_metrics_aggregation_fn=weighted_average,
        accept_failures=False,
        on_fit_config_fn=fit_config_fn,
        evaluate_fn=eval_fn,
        fraction_evaluate=0.0,
    )
    histotory= fl.simulation.start_simulation(
        client_fn=get_client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy= strategy,
        client_resources = cfg.client_resources,
    )

if __name__ == "__main__":
    main()