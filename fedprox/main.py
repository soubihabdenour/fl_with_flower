import hydra
from omegaconf import dictconfig, omegaconf, DictConfig, OmegaConf
import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn
from fedprox.server import get_on_fit_config


@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):
    # Get experement output dir from config
    print(OmegaConf.to_yaml(cfg))

    # prepare dataset

    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients,
        cfg.batch_size,
        cfg.test_batch_size,
        cfg.val_ratio)

    print(len(trainloaders), len(trainloaders[0].dataset))

    # defining client
    client_fn = generate_client_fn(trainloaders,
                                   validationloaders,
                                   cfg.num_classes
                                   )

    # defining strategy
    strategy = fl.server.strategy.FedProx(fraction_fit=0.00001,
                                          proximal_mu=cfg.proximal_mu,
                                          min_fit_clients=cfg.clients_per_round_fit,
                                          min_evaluate_clients=cfg.clients_per_round_val,
                                          min_available_clients=cfg.num_clients,
                                          on_fit_config_fn=get_on_fit_config(cfg.config_fit))

    #start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,

    )

if __name__ == '__main__':
    main()