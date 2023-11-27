from argparse import ArgumentParser

from ml_collections import FrozenConfigDict
import yaml


def read_config_from_yaml():
    parser = ArgumentParser()
    parser.add_argument("config", default="config-default.yaml", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())

    return FrozenConfigDict(config)
