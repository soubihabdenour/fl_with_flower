import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    clients_to_instantiate = []
    for client_cfg in cfg.clients:
        #print(client_cfg)
        #print(client_cfg.client)
        print(hydra.compose(config_name=f"client/{client_cfg.client}"))
    print(clients_to_instantiate)


if __name__ == "__main__":
    main()