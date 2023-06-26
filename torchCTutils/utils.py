from argparse import ArgumentParser
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def get(self, key, default_value):
        return self.__dict__.get(key, default_value)


def read_config():
    parser = ArgumentParser()
    parser.add_argument("config", default="config-default.yaml", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())

    return Config(config)
