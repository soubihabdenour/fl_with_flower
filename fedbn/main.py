import hydra
from datasets import disable_progress_bar
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from client import get_client_fn
from dataset import get_data
from fedbn.utils import NetWithBnAndFrozen
from server import fit_config, get_evaluate_fn, evaluate_metrics_aggregation_fn, get_all_model_parameters
from plot import smooth_plot


@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus,
    }
    fds, centralized_testset = get_data(partitions_number=cfg.num_clients, config=cfg.dataset)
    #model = NetWithBnAndFrozen(num_classes=cfg.model.num_classes, freeze_cnn_layer=False)
    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(fds, num_classes=cfg.model.num_classes),
        num_clients=cfg.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.5,  # Sample 10% of available clients for training
            fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
            min_available_clients=10,
            on_fit_config_fn=fit_config(cfg.config_fit),
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Aggregate federated metrics
            evaluate_fn=get_evaluate_fn(centralized_testset, cfg.model.num_classes),  # Global evaluation function
            #initial_parameters=get_all_model_parameters(model)
        ),
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )
    smooth_plot(history, "chest-xray-pneumonia - IID - 100 clients with 10 per round")


if __name__ == "__main__":
    main()
