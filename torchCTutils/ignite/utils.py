from pathlib import Path

import torch
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger


def initialize_engines(model, optimizer, criterion, metrics, device):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger("Trainer")

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")

    validation_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )
    validation_evaluator.logger = setup_logger("Val Evaluator")
    return trainer, train_evaluator, validation_evaluator


def load_checkpoint(resume_from, to_save):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
