import json
from pathlib import Path


def clearml_scalars_parser(path: Path, ax):
    with open(path, "r") as f:
        scalars = json.load(f)

    for scalar in scalars:
        ax.plot(scalar["x"], scalar["y"], label=scalar["name"])
    ax.legend(loc="best")

    return ax
