"""Streaming dataset support — lazy iteration over large datasets.

Provides a ``StreamingDataset`` class that iterates over data sources
without loading everything into memory. Pure Python, no external deps.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)


class StreamingDataset:
    """Lazily iterate over a data source without loading into memory.

    *source* can be a file path (string or ``Path``) or any iterable that
    yields examples. When a file path is given, the file is (re-)opened on
    every iteration so the stream is replayable.
    """

    def __init__(
        self,
        source: Union[str, Path, Iterable[Any]],
        format: str = "jsonl",
    ) -> None:
        self._path: Optional[Path] = None
        self._iterable: Optional[Iterable[Any]] = None
        self._format = format
        self._transforms: list[tuple[str, Any]] = []

        if isinstance(source, (str, Path)):
            self._path = Path(source)
        else:
            self._iterable = source

    # ── internal helpers ────────────────────────────────────────────────── #

    @staticmethod
    def _gen_filter(it: Iterator[Any], pred: Callable[[Any], bool]) -> Iterator[Any]:
        for x in it:
            if pred(x):
                yield x

    @staticmethod
    def _gen_map(it: Iterator[Any], fn: Callable[[Any], Any]) -> Iterator[Any]:
        for x in it:
            yield fn(x)

    def _raw_iter(self) -> Iterator[Any]:
        """Yield raw items from the underlying source."""
        if self._path is not None:
            with open(self._path, "r", encoding="utf-8") as fh:
                if self._format == "jsonl":
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
                else:
                    for line in fh:
                        yield line.rstrip("\n")
        elif self._iterable is not None:
            yield from self._iterable
        else:
            return

    def _apply_transforms(self, it: Iterator[Any]) -> Iterator[Any]:
        """Apply chained transforms to an iterator."""
        for kind, arg in self._transforms:
            if kind == "filter":
                it = self._gen_filter(it, arg)
            elif kind == "map":
                it = self._gen_map(it, arg)
            elif kind == "skip":
                n = arg
                count = 0
                items = []
                for x in it:
                    if count < n:
                        count += 1
                        continue
                    items.append(x)
                it = iter(items)
            elif kind == "take":
                n = arg
                items = []
                count = 0
                for x in it:
                    if count >= n:
                        break
                    items.append(x)
                    count += 1
                it = iter(items)
        return it

    # ── public API ──────────────────────────────────────────────────────── #

    def __iter__(self) -> Iterator[Any]:
        return self._apply_transforms(self._raw_iter())

    def take(self, n: int) -> "StreamingDataset":
        """Return a new ``StreamingDataset`` with at most *n* examples."""
        new = StreamingDataset.__new__(StreamingDataset)
        new._path = self._path
        new._iterable = self._iterable
        new._format = self._format
        new._transforms = list(self._transforms) + [("take", n)]
        return new

    def skip(self, n: int) -> "StreamingDataset":
        """Return a new ``StreamingDataset`` that skips the first *n* examples."""
        new = StreamingDataset.__new__(StreamingDataset)
        new._path = self._path
        new._iterable = self._iterable
        new._format = self._format
        new._transforms = list(self._transforms) + [("skip", n)]
        return new

    def filter(self, predicate: Callable[[Any], bool]) -> "StreamingDataset":
        """Return a new ``StreamingDataset`` keeping only matching examples."""
        new = StreamingDataset.__new__(StreamingDataset)
        new._path = self._path
        new._iterable = self._iterable
        new._format = self._format
        new._transforms = list(self._transforms) + [("filter", predicate)]
        return new

    def map(self, fn: Callable[[Any], Any]) -> "StreamingDataset":
        """Return a new ``StreamingDataset`` with *fn* applied to each element."""
        new = StreamingDataset.__new__(StreamingDataset)
        new._path = self._path
        new._iterable = self._iterable
        new._format = self._format
        new._transforms = list(self._transforms) + [("map", fn)]
        return new

    def count(self) -> int:
        """Count total examples. **Consumes the iterator.**"""
        n = 0
        for _ in self:
            n += 1
        return n


def stream_jsonl(path: Union[str, Path]) -> StreamingDataset:
    """Convenience: create a ``StreamingDataset`` for a JSONL file."""
    return StreamingDataset(path, format="jsonl")


def stream_interleave(
    datasets: Sequence[StreamingDataset],
    weights: Optional[Sequence[float]] = None,
    *,
    seed: Optional[int] = None,
) -> StreamingDataset:
    """Interleave multiple streaming datasets by weight.

    When *weights* is ``None``, uniform weights are used. The interleaving
    is probabilistic: at each step a dataset is chosen proportionally to
    its weight, and the next element from that dataset is yielded.

    Returns a ``StreamingDataset`` wrapping the interleaved iterable.
    """
    if not datasets:
        return StreamingDataset(iter([]))

    n = len(datasets)

    if weights is None:
        weights_list = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(
                f"weights length ({len(weights)}) != datasets length ({n})"
            )
        total = sum(weights)
        if total <= 0:
            raise ValueError("Sum of weights must be > 0")
        weights_list = [w / total for w in weights]

    def _interleave() -> Iterator[Any]:
        rng = random.Random(seed)
        iterators = [iter(ds) for ds in datasets]
        exhausted = [False] * n

        while not all(exhausted):
            # Build cumulative weights for non-exhausted iterators
            active_indices = [i for i in range(n) if not exhausted[i]]
            if not active_indices:
                break
            active_weights = [weights_list[i] for i in active_indices]
            total_w = sum(active_weights)
            if total_w <= 0:
                break

            # Weighted random choice
            r = rng.random() * total_w
            cumulative = 0.0
            chosen_idx = active_indices[0]
            for ai, aw in zip(active_indices, active_weights):
                cumulative += aw
                if r <= cumulative:
                    chosen_idx = ai
                    break

            try:
                yield next(iterators[chosen_idx])
            except StopIteration:
                exhausted[chosen_idx] = True

    return StreamingDataset(_interleave())
