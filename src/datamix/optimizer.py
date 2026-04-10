"""Automatic dataset ratio optimization."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from datamix._types import DatamixError, DatasetProfile


@dataclass
class OptimizationResult:
    """Result of a ratio optimization run."""

    weights: Dict[str, float]
    score: float
    iterations: int
    converged: bool


# ── Objectives ────────────────────────────────────────────────────────

_VALID_OBJECTIVES = {"balanced", "diversity", "quality"}


def _obj_balanced(weights: Dict[str, float], profiles: Dict[str, DatasetProfile]) -> float:
    """Equal quality contribution: minimise variance of (weight * quality)."""
    contributions = []
    for name, w in weights.items():
        q = profiles[name].quality.avg_quality_score or 1.0
        contributions.append(w * q)
    if not contributions:
        return 0.0
    mean = sum(contributions) / len(contributions)
    var = sum((c - mean) ** 2 for c in contributions) / len(contributions)
    # Score: 1 at zero variance, decaying toward 0.
    return 1.0 / (1.0 + var * len(contributions) * 100)


def _obj_diversity(weights: Dict[str, float], profiles: Dict[str, DatasetProfile]) -> float:
    """Maximize vocabulary / category diversity across the mix."""
    all_cats: Dict[str, float] = {}
    for name, w in weights.items():
        for cat, count in profiles[name].categories.items():
            all_cats[cat] = all_cats.get(cat, 0.0) + w * count
    if not all_cats:
        # Fall back to entropy over weights themselves
        vals = [v for v in weights.values() if v > 0]
        if not vals:
            return 0.0
        total = sum(vals)
        entropy = -sum((v / total) * math.log(v / total + 1e-12) for v in vals)
        max_ent = math.log(max(len(vals), 1))
        return entropy / max_ent if max_ent > 0 else 0.0

    total = sum(all_cats.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in all_cats.values() if v > 0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_ent = math.log(max(len(probs), 1))
    return entropy / max_ent if max_ent > 0 else 0.0


def _obj_quality(weights: Dict[str, float], profiles: Dict[str, DatasetProfile]) -> float:
    """Maximize weighted mean quality score."""
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    score = sum(
        w * (profiles[name].quality.avg_quality_score or 0.0)
        for name, w in weights.items()
    )
    return score / total_w


_OBJECTIVES = {
    "balanced": _obj_balanced,
    "diversity": _obj_diversity,
    "quality": _obj_quality,
}


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n else {}
    return {k: v / total for k, v in weights.items()}


def _apply_budget(weights: Dict[str, float], profiles: Dict[str, DatasetProfile], budget: int) -> Dict[str, float]:
    """Clip weights so no source contributes more tokens than it has."""
    result = dict(weights)
    for name, w in result.items():
        max_w = profiles[name].n_tokens / budget if budget > 0 else 1.0
        result[name] = min(w, max_w)
    return _normalize(result)


# ── Public API ────────────────────────────────────────────────────────


def optimize_ratios(
    profiles: Dict[str, DatasetProfile],
    objective: str = "balanced",
    token_budget: Optional[int] = None,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> OptimizationResult:
    """Optimize dataset mixing ratios for a given objective.

    Uses iterative coordinate-wise adjustment (gradient-free).

    Args:
        profiles: Mapping of source name to DatasetProfile.
        objective: One of "balanced", "diversity", "quality".
        token_budget: Optional total token budget.
        max_iterations: Maximum optimisation steps.
        tolerance: Convergence threshold on score change.
    """
    if objective not in _VALID_OBJECTIVES:
        raise DatamixError(f"Unknown objective {objective!r}, expected one of {_VALID_OBJECTIVES}")
    if not profiles:
        return OptimizationResult(weights={}, score=0.0, iterations=0, converged=True)

    obj_fn = _OBJECTIVES[objective]
    names = sorted(profiles.keys())
    n = len(names)

    # Start with equal weights
    weights = {name: 1.0 / n for name in names}
    if token_budget:
        weights = _apply_budget(weights, profiles, token_budget)

    best_score = obj_fn(weights, profiles)
    step = 0.05  # coordinate step size

    converged = False
    it = 0
    for it in range(1, max_iterations + 1):
        improved = False
        for name in names:
            for direction in (step, -step):
                trial = dict(weights)
                trial[name] = max(0.0, trial[name] + direction)
                trial = _normalize(trial)
                if token_budget:
                    trial = _apply_budget(trial, profiles, token_budget)
                s = obj_fn(trial, profiles)
                if s > best_score + tolerance:
                    weights = trial
                    best_score = s
                    improved = True
        if not improved:
            converged = True
            break

    weights = {k: round(v, 6) for k, v in _normalize(weights).items()}

    return OptimizationResult(
        weights=weights,
        score=round(best_score, 6),
        iterations=it,
        converged=converged,
    )


def grid_search_ratios(
    profiles: Dict[str, DatasetProfile],
    objective: str = "balanced",
    steps: int = 10,
) -> List[OptimizationResult]:
    """Try all ratio combinations at the given granularity and rank them.

    For *n* sources and *steps* granularity the search space is
    C(steps + n - 1, n - 1) combinations.

    Args:
        profiles: Mapping of source name to DatasetProfile.
        objective: One of "balanced", "diversity", "quality".
        steps: Number of discrete steps (e.g. 10 ⇒ 0.0, 0.1, …, 1.0).
    """
    if objective not in _VALID_OBJECTIVES:
        raise DatamixError(f"Unknown objective {objective!r}, expected one of {_VALID_OBJECTIVES}")
    if not profiles:
        return []

    obj_fn = _OBJECTIVES[objective]
    names = sorted(profiles.keys())
    n = len(names)

    results: List[OptimizationResult] = []
    # Generate all partitions of `steps` into `n` non-negative parts
    for combo in itertools.combinations(range(steps + n - 1), n - 1):
        parts: List[int] = []
        prev = -1
        for c in combo:
            parts.append(c - prev - 1)
            prev = c
        parts.append(steps + n - 2 - prev)

        weights = {names[i]: parts[i] / steps for i in range(n)}
        score = obj_fn(weights, profiles)
        results.append(OptimizationResult(
            weights={k: round(v, 6) for k, v in weights.items()},
            score=round(score, 6),
            iterations=0,
            converged=True,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def format_optimization(result: OptimizationResult) -> str:
    """Format an OptimizationResult as a human-readable string."""
    lines = [
        "Optimization Result",
        "=" * 40,
        f"Score     : {result.score:.6f}",
        f"Converged : {result.converged}",
        f"Iterations: {result.iterations}",
        "",
        "Weights:",
    ]
    for name in sorted(result.weights, key=lambda k: result.weights[k], reverse=True):
        w = result.weights[name]
        bar = "#" * int(w * 40)
        lines.append(f"  {name:30s} {w:6.2%}  {bar}")
    return "\n".join(lines)
