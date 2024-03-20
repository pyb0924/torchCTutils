from argparse import ArgumentParser

from ml_collections import ConfigDict
from prettytable import PrettyTable
import yaml

display_configs = [
    "model",
    "use_residual_learning",
    "lambda_adv",
    "lambda_rec",
    "lambda_proj",
    "lambda_roi",
    "lr",
    "train_batch_size",
    "val_batch_size",
    "fold",
    "gpu_ids",
]


def print_config(config: ConfigDict, display_configs=display_configs):
    table = PrettyTable()
    table.title = "Table: Train Configs"
    for key in display_configs:
        if key in config:
            table.add_column(key, [config[key]])
    print(table)


def read_config_from_yaml(display=True):
    parser = ArgumentParser()
    parser.add_argument("config", default="config-default.yaml", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())

    config = ConfigDict(config)
    if display:
        print_config(config)

    return config
