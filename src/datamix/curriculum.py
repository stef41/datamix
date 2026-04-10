"""Curriculum scheduling — phase-based training data progression."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from datamix._types import CurriculumPhase, CurriculumSchedule


def linear_schedule(
    dataset_names: List[str],
    n_phases: int = 3,
    start_weights: Optional[Dict[str, float]] = None,
    end_weights: Optional[Dict[str, float]] = None,
    total_tokens: int = 0,
) -> CurriculumSchedule:
    """Create a linear curriculum that interpolates weights from start to end.

    Args:
        dataset_names: Names of datasets to schedule.
        n_phases: Number of phases.
        start_weights: Weights at beginning (default: equal).
        end_weights: Weights at end (default: equal).
        total_tokens: Total tokens for the curriculum.
    """
    if not dataset_names:
        return CurriculumSchedule()

    n = len(dataset_names)
    if start_weights is None:
        start_weights = {name: 1.0 / n for name in dataset_names}
    if end_weights is None:
        end_weights = {name: 1.0 / n for name in dataset_names}

    phases: List[CurriculumPhase] = []
    for i in range(n_phases):
        start_frac = i / n_phases
        end_frac = (i + 1) / n_phases
        t = (i + 0.5) / n_phases  # midpoint of phase

        weights: Dict[str, float] = {}
        for name in dataset_names:
            sw = start_weights.get(name, 0.0)
            ew = end_weights.get(name, 0.0)
            weights[name] = round(sw + (ew - sw) * t, 6)

        phases.append(CurriculumPhase(
            name=f"phase_{i}",
            start_fraction=round(start_frac, 6),
            end_fraction=round(end_frac, 6),
            weights=weights,
        ))

    return CurriculumSchedule(
        name="linear",
        phases=phases,
        total_tokens=total_tokens,
    )


def cosine_schedule(
    dataset_names: List[str],
    n_phases: int = 4,
    primary: Optional[str] = None,
    total_tokens: int = 0,
) -> CurriculumSchedule:
    """Create a cosine-annealed curriculum.

    The primary dataset starts high and cosine-decays; others increase.

    Args:
        dataset_names: Names of datasets.
        n_phases: Number of phases.
        primary: Primary dataset name (default: first).
        total_tokens: Total tokens.
    """
    if not dataset_names:
        return CurriculumSchedule()

    primary = primary or dataset_names[0]
    others = [n for n in dataset_names if n != primary]

    phases: List[CurriculumPhase] = []
    for i in range(n_phases):
        start_frac = i / n_phases
        end_frac = (i + 1) / n_phases
        t = (i + 0.5) / n_phases

        # Cosine decay for primary
        primary_w = 0.5 * (1 + math.cos(math.pi * t))
        remaining = 1.0 - primary_w

        weights: Dict[str, float] = {primary: round(primary_w, 6)}
        if others:
            each = remaining / len(others)
            for name in others:
                weights[name] = round(each, 6)

        phases.append(CurriculumPhase(
            name=f"phase_{i}",
            start_fraction=round(start_frac, 6),
            end_fraction=round(end_frac, 6),
            weights=weights,
        ))

    return CurriculumSchedule(
        name="cosine",
        phases=phases,
        total_tokens=total_tokens,
    )


def step_schedule(
    phases_config: List[Dict[str, object]],
    total_tokens: int = 0,
) -> CurriculumSchedule:
    """Create a step-function curriculum from explicit phase configs.

    Args:
        phases_config: List of dicts with 'name', 'fraction', 'weights' keys.
            fraction is the duration of this phase (sums to 1.0).
        total_tokens: Total tokens.

    Example:
        step_schedule([
            {"name": "warmup", "fraction": 0.1, "weights": {"wiki": 0.8, "code": 0.2}},
            {"name": "main", "fraction": 0.7, "weights": {"wiki": 0.5, "code": 0.5}},
            {"name": "cooldown", "fraction": 0.2, "weights": {"wiki": 0.3, "code": 0.7}},
        ])
    """
    phases: List[CurriculumPhase] = []
    cursor = 0.0

    for cfg in phases_config:
        name = str(cfg.get("name", f"phase_{len(phases)}"))
        fraction = float(cfg.get("fraction", 0.0))  # type: ignore[arg-type]
        weights = dict(cfg.get("weights", {}))  # type: ignore[arg-type]

        phases.append(CurriculumPhase(
            name=name,
            start_fraction=round(cursor, 6),
            end_fraction=round(cursor + fraction, 6),
            weights=weights,
        ))
        cursor += fraction

    return CurriculumSchedule(
        name="step",
        phases=phases,
        total_tokens=total_tokens,
    )


def custom_schedule(
    phases: List[CurriculumPhase],
    name: str = "custom",
    total_tokens: int = 0,
) -> CurriculumSchedule:
    """Create a curriculum from pre-built phases."""
    return CurriculumSchedule(name=name, phases=phases, total_tokens=total_tokens)
