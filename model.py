from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import termcolor
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm  # type: ignore

from concept import IntervalConcept, MultipleConcept
from dataset import NumberGameDataset


@dataclass
class SampleRun:
    given: Iterable[float]
    truth: Iterable[float]
    pred: Iterable[float]

    def print(self) -> None:
        given = [i for i, v in enumerate(self.given) if v]
        truth = [i for i, v in enumerate(self.truth) if v]
        prediction = [i for i, v in sorted(enumerate(self.pred), key=lambda z: -z[1])]

        def colorstr(i) -> str:
            if i in given:
                return termcolor.colored(str(i), "green", attrs=["bold"])
            elif i in truth:
                return termcolor.colored(str(i), "yellow", attrs=["bold"])
            else:
                return termcolor.colored(str(i), "cyan")

        print("Sample run:")
        print("\t Given: ", ", ".join(colorstr(i) for i in given))
        print("\t Truth: ", ", ".join(colorstr(i) for i in truth))
        print("\t Prediction: ", " > ".join(colorstr(i) for i in prediction))


class BayesianModel(nn.Module):
    def __init__(self, *, universe_size: int, math_prior: float, interval_prior: float) -> None:
        super().__init__()
        self.hypotheses = torch.stack(
            [
                torch.from_numpy(MultipleConcept(factor)._accept(np.arange(universe_size)))
                for factor in range(2, 6)
            ] + [
                torch.from_numpy(IntervalConcept(lo=lo, hi=lo+9)._accept(np.arange(universe_size)))
                for lo in (1, 11, 21)
            ]
        )
        self.priors = torch.stack(
            [
                torch.tensor(math_prior / 4)
                for factor in range(2, 6)
            ] + [
                torch.tensor(interval_prior / 3)
                for lo in (1, 11, 21)
            ]
        )
        self.hypotheses[:, 0] = 0  # Disallow

    @torch.no_grad()
    def forward(self, given: torch.LongTensor) -> torch.Tensor:
        x = given[..., None, :]
        log_likelihood = torch.where(
            (x > self.hypotheses).sum(dim=-1) == 0,
            -x.sum(dim=-1) * torch.log(self.hypotheses.sum(dim=-1)),
            -float("inf"),
        )
        posterior = F.softmax(torch.log(self.priors) + log_likelihood, dim=-1)
        return (posterior[..., None] * self.hypotheses).sum(dim=-2)


class NumberGameModel(nn.Module):
    def __init__(self, *, universe_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.key = nn.Embedding(universe_size, embedding_dim)
        self.query = nn.Embedding(universe_size, embedding_dim)
        self.universe_size = universe_size
        self.embedding_dim = embedding_dim

    def forward(self, given: torch.LongTensor) -> torch.Tensor:
        z = torch.einsum("...i,iz,jz->...j", given.float(), self.key.weight, self.query.weight)
        return F.softmax(z, dim=-1)


def train(
    *,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    data: torch.utils.data.DataLoader,
) -> float:
    model.train()
    pbar = tqdm(data)
    total_loss = 0.0
    cnt = 0
    for i, (given, truth) in enumerate(pbar):
        pred = model(given)
        loss = F.binary_cross_entropy(pred[truth == 1], truth[truth == 1].float(), reduction="none")
        total_loss += loss.sum().item()
        cnt += truth[truth == 1].numel()
        loss = loss.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"Training loss: {loss.item():.4f}")

    return total_loss / cnt


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    data: torch.utils.data.DataLoader,
) -> tuple[float, SampleRun]:
    model.eval()
    loss = 0.0
    cnt = 0
    for i, (given, truth) in enumerate(data):
        pred = model(given)
        loss += F.binary_cross_entropy(pred[truth == 1], truth[truth == 1].float(), reduction="sum").item()
        cnt += truth[truth == 1].numel()
    idx = np.random.randint(len(given))
    sample_run = SampleRun(
        given=given[idx],
        truth=truth[idx],
        pred=pred[idx],
    )
    return loss / cnt, sample_run


def run(
    *,
    max_val: int,
    embedding_dim: int,
    train_ds: NumberGameDataset,
    eval_ds: NumberGameDataset,
    n_epoch: int,
    lr: float,
) -> tuple[NumberGameModel, list[dict[str, float]]]:
    model = NumberGameModel(
        universe_size=max_val+1,
        embedding_dim=embedding_dim,
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=True)
    eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=1024, shuffle=False)

    losses: list[dict[str, float]] = []

    for i in range(n_epoch):
        termcolor.cprint("="*shutil.get_terminal_size().columns, attrs=["bold"])
        termcolor.cprint(f"Epoch {i+1} of {n_epoch}", "magenta", attrs=["bold"])
        train_loss = train(model=model, optim=optim, data=train_dl)
        eval_loss, sample_run = evaluate(model=model, data=eval_dl)

        losses.append({"train": train_loss, "eval": eval_loss})
        print("Evaluation loss:", eval_loss)
        sample_run.print()

        # For illustration's sake, also do this sample run:
        given = torch.LongTensor([[i == 12 for i in range(max_val+1)]])
        SampleRun(
            given=given[0],
            truth=torch.zeros_like(given[0]),
            pred=model(given)[0],
        ).print()

    return model, losses
