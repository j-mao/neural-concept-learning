from __future__ import annotations

import functools
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import Random
from typing import Generator, Iterator

import numpy as np


class BaseConcept(ABC):
    def generate(self, *, lo: int = 1, hi: int = 30, seed: int) -> Generator[int, None, None]:
        rng = Random(seed)
        yield from (
            num
            for num in rng.sample(range(lo, hi+1), hi-lo+1)
            if self._accept(num, rng=rng)
        )

    @abstractmethod
    def _accept(self, num: int, **kwargs) -> bool | np.ndarray:
        ...

    def __and__(self, oth: BaseConcept) -> BaseConcept:
        return AndConcept(self, oth)


class AndConcept(BaseConcept):
    def __init__(self, a: BaseConcept, b: BaseConcept) -> None:
        self._a = a
        self._b = b

    def _accept(self, num: int, **kwargs) -> bool | np.ndarray:
        return self._a._accept(num, **kwargs) and self._b._accept(num, **kwargs)


class MultipleConcept(BaseConcept):
    def __init__(self, factor: int) -> None:
        self._factor = factor

    def _accept(self, num: int | np.ndarray, **kwargs) -> bool | np.ndarray:
        return num % self._factor == 0


class IntervalConcept(BaseConcept):
    def __init__(self, lo: int, hi: int) -> None:
        self._lo = lo
        self._hi = hi

    def _accept(self, num: int | np.ndarray, **kwargs) -> bool | np.ndarray:
        return (self._lo <= num) & (num <= self._hi)


class NonsenseConcept(BaseConcept):
    def __init__(self, p_acc: float) -> None:
        self._p = p_acc

    @functools.lru_cache(maxsize=None)
    def _accept(self, num: int, rng: Random, **kwargs) -> bool:
        return rng.random() < self._p


@dataclass
class Prompt:
    concept: BaseConcept
    qty: int
    seed: int

    def __iter__(self) -> Iterator[int]:
        return itertools.islice(self.concept.generate(seed=self.seed), self.qty)

    def truth(self) -> Generator[int, None, None]:
        yield from self.concept.generate(seed=self.seed)

    def __hash__(self) -> int:
        return hash(tuple(self))


class PromptGenerator:
    def __init__(self) -> None:
        self.prompt_classes: list[tuple[list[Prompt], int]] = []

    def __len__(self) -> int:
        return sum(qty for prompt_class, qty in self.prompt_classes)

    def register(self, qty: int, prompt_class: list[Prompt]) -> None:
        self.prompt_classes.append((prompt_class, qty))

    def generate(self, seed: int) -> list[dict]:
        rng = Random(seed)
        return [
            {
                "id": hex(hash(prompt))[-7:],
                "prompt": list(prompt),
                "truth": list(prompt.truth()),
            }
            for prompt_class, qty in self.prompt_classes
            for prompt in rng.sample(prompt_class, qty)
        ]


def main() -> None:
    gen = PromptGenerator()
    for i in range(2, 6):
        qty = 4 if i < 4 else 3
        gen.register(
            6,
            [
                Prompt(concept=MultipleConcept(i), qty=qty, seed=seed)
                for seed in range(10, 16)
            ],
        )
    for i in range(3):
        gen.register(
            3,
            [
                Prompt(concept=IntervalConcept(10*i+1, 10*(i+1)), qty=4, seed=seed)
                for seed in range(20, 23)
            ],
        )
    gen.register(
        6,
        [
            Prompt(concept=NonsenseConcept(0.25), qty=3, seed=i)
            for i in range(30, 36)
        ]
    )
    # for i in range(3):
    #     gen.register(
    #         2,
    #         [
    #             Prompt(concept=MultipleConcept(2) & IntervalConcept(10*i+1, 10*(i+1)), qty=3, seed=seed)
    #             for seed in range(40, 42)
    #         ]
    #     )
    for s in gen.generate(1):
        print(s)

if __name__ == "__main__":
    main()
