from __future__ import annotations

import os

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import torch
from torch.nn import functional as F

from model import BayesianModel


def plot_learning_curve(
    data_source: str,
    embedding_dims: list[int],
    fname: str,
) -> None:
    df = pd.concat({
        embedding_dim: pd.read_json(os.path.join("output", f"loss_{data_source}_{embedding_dim}.json"))
        for embedding_dim in embedding_dims
    })
    df.index.names = ["embedding_dim", "epoch / 128"]  # We have 128 reps per epoch.
    df = df.reset_index()
    df["epoch"] = (df["epoch / 128"] + 1) * 128

    plt.figure()
    xticks = np.concatenate((np.arange(100, 1000, 100), np.arange(1000, 11000, 1000)))
    g = sns.lineplot(df, x="epoch", y="train", hue="embedding_dim", palette="Spectral")
    g.set(xscale="log", xticks=xticks, xlabel="Epochs", ylabel="Training loss")
    plt.savefig(os.path.join("figures", fname), bbox_inches="tight")
    plt.close()


@torch.no_grad()
def plot_prediction(
    data_source: str,
    embedding_dims: list[int],
    max_val: int,
    given: list[int],
    pred_fname: str,
    corr_fname: str,
) -> None:
    given_t = F.one_hot(torch.LongTensor(given), num_classes=max_val+1).sum(dim=-2).unsqueeze(0)
    bayesian = BayesianModel(universe_size=max_val+1, math_prior=0.5, interval_prior=0.5)

    df = pd.DataFrame(
        {
            "embedding_dim": embedding_dim,
            "num": i,
            "score": score.item(),
        }
        for embedding_dim in embedding_dims
        for i, score in enumerate(torch.load(os.path.join("output", f"model_{data_source}_{embedding_dim}.pth"))(given_t)[0])
    )
    df = pd.concat((
        df,
        pd.DataFrame(
            {
                "embedding_dim": -1,
                "num": i,
                "score": score.item()
            }
            for i, score in enumerate(bayesian(given_t)[0])
        )
    ))
    df["score"] = df.groupby("embedding_dim")["score"].transform(lambda x: x / x.max())

    plt.figure()
    g = sns.catplot(
        df[df["num"] > 0],
        x="num",
        y="score",
        row="embedding_dim",
        kind="bar",
        aspect=5,
        height=1,
        palette=list(map(lambda x: "tab:red" if x else "tab:blue", given_t[0,1:])),
    )
    g.set_titles("Attention model with latent dimension {row_name}")
    g.axes[0, 0].set_title("Bayesian model with 0.5 math and 0.5 interval prior")
    g.set(xticks=[9, 19, 29])
    plt.savefig(os.path.join("figures", pred_fname), bbox_inches="tight")
    plt.close()

    plt.figure()
    corr = df.set_index(["num", "embedding_dim"]).unstack("embedding_dim")["score"].corr()[-1]
    corr[corr.index > 0].reset_index().plot.scatter(x="embedding_dim", y=-1, figsize=(6, 4))
    plt.xlabel("Latent dimension")
    plt.ylabel("Correlation coefficient with Bayesian model")
    plt.savefig(os.path.join("figures", corr_fname), bbox_inches="tight")
    plt.close()


@torch.no_grad()
def plot_weights(
    data_source: str,
    embedding_dim: int,
    query_fname: str,
    key_fname: str,
) -> None:
    model = torch.load(os.path.join("output", f"model_{data_source}_{embedding_dim}.pth"))

    plt.figure()
    g = sns.heatmap(model.query.weight[1:].numpy(), cmap="RdYlBu_r", square=True, vmin=-2.5, vmax=2.5)
    g.set(xlabel="dimension", ylabel="number", xticklabels=1+np.arange(embedding_dim), yticks=np.arange(0.5, 30.5, 2), yticklabels=np.arange(1, 31, 2))
    plt.savefig(os.path.join("figures", query_fname), bbox_inches="tight")
    plt.close()

    plt.figure()
    g = sns.heatmap(model.key.weight[1:].numpy(), cmap="RdYlBu_r", square=True, vmin=-2.5, vmax=2.5)
    g.set(xlabel="dimension", ylabel="number", xticklabels=1+np.arange(embedding_dim), yticks=np.arange(0.5, 30.5, 2), yticklabels=np.arange(1, 31, 2))
    plt.savefig(os.path.join("figures", key_fname), bbox_inches="tight")
    plt.close()


def main(embedding_dims: list[int]) -> None:
    os.makedirs("figures", exist_ok=True)
    for data_source in ("none", "data_csv"):
        plot_learning_curve(
            data_source=data_source,
            embedding_dims=embedding_dims,
            fname=f"{data_source}_learning_curve.png",
        )
        for given in (
            [12],
            [12, 16],
            [1, 5, 13],
        ):
            plot_prediction(
                data_source=data_source,
                embedding_dims=embedding_dims,
                max_val=30,
                given=given,
                pred_fname=f"{data_source}_prediction_{'_'.join(map(str, given))}.png",
                corr_fname=f"{data_source}_corr_{'_'.join(map(str, given))}.png",
            )
        for embedding_dim in embedding_dims:
            plot_weights(
                data_source=data_source,
                embedding_dim=embedding_dim,
                query_fname=f"{data_source}_{embedding_dim}_weights_query.png",
                key_fname=f"{data_source}_{embedding_dim}_weights_key.png",
            )


if __name__ == "__main__":
    main(embedding_dims=[1, 2, 3, 5, 7, 10])
