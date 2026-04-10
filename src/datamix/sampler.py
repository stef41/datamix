"""Sampling strategies for dataset mixing."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Sequence

from datamix._types import MixRecipe, SamplingConfig


def temperature_sample(
    datasets: Dict[str, List[str]],
    n_samples: int,
    temperature: float = 1.0,
    seed: int = 42,
) -> List[str]:
    """Sample from multiple datasets using temperature scaling.

    Temperature > 1 makes sampling more uniform across datasets.
    Temperature < 1 amplifies differences (larger datasets get even more).
    Temperature = 1 is proportional to dataset size.

    Args:
        datasets: Mapping of dataset name → list of examples.
        n_samples: Total number of samples to draw.
        temperature: Sampling temperature.
        seed: Random seed.
    """
    rng = random.Random(seed)

    if not datasets:
        return []

    # Compute temperature-scaled weights
    sizes = {name: len(examples) for name, examples in datasets.items()}
    raw = {name: max(1, s) ** (1.0 / max(0.01, temperature)) for name, s in sizes.items()}
    total = sum(raw.values())
    weights = {name: v / total for name, v in raw.items()}

    # Sample
    result: List[str] = []
    names = list(datasets.keys())
    probs = [weights[n] for n in names]

    for _ in range(n_samples):
        # Weighted random choice
        r = rng.random()
        cumulative = 0.0
        chosen = names[0]
        for name, p in zip(names, probs):
            cumulative += p
            if r <= cumulative:
                chosen = name
                break

        pool = datasets[chosen]
        if pool:
            result.append(rng.choice(pool))

    return result


def proportional_sample(
    datasets: Dict[str, List[str]],
    n_samples: int,
    seed: int = 42,
) -> List[str]:
    """Sample proportionally to dataset sizes."""
    return temperature_sample(datasets, n_samples, temperature=1.0, seed=seed)


def stratified_sample(
    examples: List[str],
    category_fn: Any,
    n_samples: int,
    seed: int = 42,
) -> List[str]:
    """Sample with equal representation from each category.

    Args:
        examples: List of text examples.
        category_fn: Function that maps text → category string.
        n_samples: Total samples to draw.
        seed: Random seed.
    """
    rng = random.Random(seed)

    # Group by category
    buckets: Dict[str, List[str]] = {}
    for ex in examples:
        cat = category_fn(ex)
        if cat not in buckets:
            buckets[cat] = []
        buckets[cat].append(ex)

    if not buckets:
        return []

    # Equal samples per category
    per_cat = max(1, n_samples // len(buckets))
    result: List[str] = []

    for cat_examples in buckets.values():
        k = min(per_cat, len(cat_examples))
        result.extend(rng.sample(cat_examples, k))

    # If we need more, sample randomly from remainder
    while len(result) < n_samples:
        all_examples = [ex for pool in buckets.values() for ex in pool]
        if not all_examples:
            break
        result.append(rng.choice(all_examples))

    return result[:n_samples]
