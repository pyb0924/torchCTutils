from argparse import ArgumentParser
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Mapping, Optional, Union
import yaml

import torch
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Engine
import ignite.distributed as idist
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger


def initialize_supervised_engines(model, optimizer, criterion, metrics, device):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger("Trainer")

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")

    validation_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )
    validation_evaluator.logger = setup_logger("Val Evaluator")
    return trainer, train_evaluator, validation_evaluator


def setup_parser():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f.read())

    parser = ArgumentParser()
    parser.add_argument("--backend", default=None, type=str)
    for k, v in config.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true")
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))

    return parser


def resume_from(
    to_load: Mapping,
    checkpoint_fp: Union[str, Path],
    logger: Logger,
    strict: bool = True,
    model_dir: Optional[str] = None,
) -> None:
    """Loads state dict from a checkpoint file to resume the training.

    Parameters
    ----------
    to_load
        a dictionary with objects, e.g. {“model”: model, “optimizer”: optimizer, ...}
    checkpoint_fp
        path to the checkpoint file
    logger
        to log info about resuming from a checkpoint
    strict
        whether to strictly enforce that the keys in `state_dict` match the keys
        returned by this module’s `state_dict()` function. Default: True
    model_dir
        directory in which to save the object
    """
    if isinstance(checkpoint_fp, str) and checkpoint_fp.startswith("https://"):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_fp,
            model_dir=model_dir,
            map_location="cpu",
            check_hash=True,
        )
    else:
        if isinstance(checkpoint_fp, str):
            checkpoint_fp = Path(checkpoint_fp)

        if not checkpoint_fp.exists():
            raise FileNotFoundError(f"Given {str(checkpoint_fp)} does not exist.")
        checkpoint = torch.load(checkpoint_fp, map_location="cpu")

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=strict)
    logger.info("Successfully resumed from a checkpoint: %s", checkpoint_fp)


def setup_output_dir(config: Any, rank: int) -> Path:
    """Create output folder."""
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{now}-backend-{config.backend}-lr-{config.lr}"
        path = Path(config.output_dir, name)
        path.mkdir(parents=True, exist_ok=True)
        config.output_dir = path.as_posix()

    return Path(idist.broadcast(config.output_dir, src=0))
