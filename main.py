from __future__ import annotations

import argparse
import json
import os
import random
from typing import Optional

import numpy as np
import torch

from dataset import NumberGameDataset, NumberGameDummyDataset, NumberGameSurveyDataset
from model import run


def main(max_val: int, embedding_dim: int, fname: Optional[str]) -> None:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    train_ds: NumberGameDataset
    eval_ds: NumberGameDataset

    if fname is None:
        train_ds = NumberGameDummyDataset(size=1024, universe_size=max_val+1, reps=128)
        eval_ds = NumberGameDummyDataset(size=512, universe_size=max_val+1, reps=1)
    else:
        train_ds = NumberGameSurveyDataset(fname=fname, nonsense="drop", universe_size=max_val+1, reps=128)
        eval_ds = NumberGameSurveyDataset(fname=fname, nonsense="keep", universe_size=max_val+1, reps=1)

    model, losses = run(
        max_val=max_val,
        embedding_dim=embedding_dim,
        train_ds=train_ds,
        eval_ds=eval_ds,
        n_epoch=80,  # Note: the datasets have 128 reps, so it's effectively 80x128 epochs.
        lr=3e-4,
    )

    tag = "{}_{}".format((fname or 'none').replace('.', '_'), embedding_dim)
    os.makedirs("output", exist_ok=True)
    torch.save(model, os.path.join("output", f"model_{tag}.pth"))
    with open(os.path.join("output", f"loss_{tag}.json"), "w") as f:
        json.dump(losses, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--max-val",
        default=30, required=False, type=int,
        help="Maximum value in the universe.",
    )
    parser.add_argument(
        "-e", "--embedding-dim",
        required=True, type=int,
        help="Dimension of the latent space.",
    )
    parser.add_argument(
        "-f", "--fname",
        default=None, required=False, type=str,
        help="Filename to load survey answers from.",
    )
    args = parser.parse_args()

    main(
        max_val=args.max_val,
        embedding_dim=args.embedding_dim,
        fname=args.fname,
    )
