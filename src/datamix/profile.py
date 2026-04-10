"""Profile datasets — compute statistics, token counts, quality metrics."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from datamix._types import DatamixError, DatasetProfile, QualityMetrics


def profile_dataset(
    examples: Sequence[str],
    name: str = "dataset",
    tokenizer: Optional[Callable[[str], List[Any]]] = None,
    category_fn: Optional[Callable[[str], str]] = None,
) -> DatasetProfile:
    """Profile a dataset from a list of text examples.

    Args:
        examples: List of text strings.
        name: Dataset name.
        tokenizer: Optional tokenizer function (text → token list).
                   If None, uses whitespace splitting as approximation.
        category_fn: Optional function to categorize examples.
    """
    if not examples:
        return DatasetProfile(name=name)

    lengths: List[int] = []
    token_counts: List[int] = []
    categories: Dict[str, int] = {}

    for text in examples:
        lengths.append(len(text))
        if tokenizer:
            tokens = tokenizer(text)
            token_counts.append(len(tokens))
        else:
            # Approximate: ~4 chars per token (GPT-like)
            token_counts.append(max(1, len(text) // 4))

        if category_fn:
            cat = category_fn(text)
            categories[cat] = categories.get(cat, 0) + 1

    n = len(examples)
    total_tokens = sum(token_counts)
    sorted_lengths = sorted(lengths)
    median_idx = n // 2

    quality = QualityMetrics(
        avg_length=statistics.mean(lengths),
        median_length=float(sorted_lengths[median_idx]),
        min_length=min(lengths),
        max_length=max(lengths),
        std_length=statistics.stdev(lengths) if n > 1 else 0.0,
    )

    return DatasetProfile(
        name=name,
        n_examples=n,
        n_tokens=total_tokens,
        avg_tokens_per_example=total_tokens / n,
        quality=quality,
        categories=categories,
    )


def profile_jsonl(
    path: str | Path,
    text_key: str = "text",
    name: Optional[str] = None,
    tokenizer: Optional[Callable[[str], List[Any]]] = None,
    max_examples: Optional[int] = None,
) -> DatasetProfile:
    """Profile a JSONL file.

    Args:
        path: Path to JSONL file.
        text_key: Key to extract text from each JSON object.
        name: Dataset name (defaults to filename).
        tokenizer: Optional tokenizer function.
        max_examples: Max examples to read (None = all).
    """
    path = Path(path)
    if not path.exists():
        raise DatamixError(f"File not found: {path}")

    examples: List[str] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(obj, dict):
                text = obj.get(text_key, "")
            elif isinstance(obj, str):
                text = obj
            else:
                continue

            if text:
                examples.append(str(text))

    dname = name or path.stem
    return profile_dataset(examples, name=dname, tokenizer=tokenizer)


def compare_profiles(profiles: List[DatasetProfile]) -> Dict[str, Any]:
    """Compare multiple dataset profiles side by side."""
    if not profiles:
        return {"datasets": [], "total_tokens": 0}

    rows = []
    for p in profiles:
        rows.append({
            "name": p.name,
            "n_examples": p.n_examples,
            "n_tokens": p.n_tokens,
            "size_tokens_m": round(p.size_tokens_m, 2),
            "avg_tokens_per_example": round(p.avg_tokens_per_example, 1),
            "avg_length": round(p.quality.avg_length, 1),
            "duplicate_rate": round(p.quality.duplicate_rate, 4),
        })

    total_tokens = sum(p.n_tokens for p in profiles)
    total_examples = sum(p.n_examples for p in profiles)

    return {
        "datasets": rows,
        "total_tokens": total_tokens,
        "total_examples": total_examples,
        "n_datasets": len(profiles),
    }
