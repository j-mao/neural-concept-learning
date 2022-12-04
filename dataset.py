from __future__ import annotations

import pandas as pd  # type: ignore
import torch
from torch.nn import functional as F

from concept import IntervalConcept, MultipleConcept, Prompt, PromptGenerator


class NumberGameDataset(torch.utils.data.Dataset):
    data: list[dict]

    def __init__(self, universe_size: int, reps: int) -> None:
        super().__init__()
        self.universe_size = universe_size
        self.reps = reps

    def __len__(self) -> int:
        return len(self.data) * self.reps

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        pt = self.data[idx % len(self.data)]
        return (
            F.one_hot(torch.LongTensor(pt["prompt"]), num_classes=self.universe_size).sum(dim=-2),
            F.one_hot(torch.LongTensor(pt["truth"]), num_classes=self.universe_size).sum(dim=-2),
        )


class NumberGameDummyDataset(NumberGameDataset):
    def __init__(self, size: int, universe_size: int, reps: int) -> None:
        super().__init__(universe_size=universe_size, reps=reps)
        self.size = size

        self.gen = PromptGenerator()
        for i in range(2, 6):
            qty = 4 if i < 4 else 3
            self.gen.register(
                size // 7,
                [
                    Prompt(concept=MultipleConcept(i), qty=qty, seed=seed)
                    for seed in range(size // 7)
                ],
            )
        for i in range(3):
            self.gen.register(
                size // 7,
                [
                    Prompt(concept=IntervalConcept(10*i+1, 10*(i+1)), qty=4, seed=seed)
                    for seed in range(size // 7)
                ],
            )
        self.data = self.gen.generate(1)


def parse(raw: str, delimiter: str = ",") -> set[int]:
    return set(map(int, raw.split(delimiter)))


class NumberGameSurveyDataset(NumberGameDataset):
    def __init__(self, fname: str, nonsense: str, universe_size: int, reps: int) -> None:
        super().__init__(universe_size=universe_size, reps=reps)
        prefix = "This machine accepts: "
        df = pd.read_csv(fname)

        columns = [column for column in df.columns if column.startswith(prefix)]
        if nonsense == "keep":
            pass
        elif nonsense == "only":
            columns = [column for i, column in enumerate(columns) if i % 9 >= 7]
        elif nonsense == "drop":
            columns = [column for i, column in enumerate(columns) if i % 9 < 7]
        else:
            raise ValueError(f"Unexpected nonsense mode: {nonsense}")

        self.data = [
            {
                "prompt": tuple(parse(column[len(prefix):])),
                "truth": tuple(parse(column[len(prefix):]) | parse(response)),
            }
            for column in columns
            for response in df[column]
            if isinstance(response, str)
        ]
