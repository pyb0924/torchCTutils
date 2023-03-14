from pathlib import Path

import torch
from ignite.handlers import Checkpoint



def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)