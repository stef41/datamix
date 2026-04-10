"""Token budget allocation — compute, fit, and report training budgets."""

from __future__ import annotations

from typing import Any, Dict, List

from datamix._types import DatasetProfile, MixRecipe, TokenBudget


def compute_budget(
    recipe: MixRecipe,
    profiles: List[DatasetProfile],
) -> TokenBudget:
    """Compute token allocations from a recipe and profiles.

    Distributes the recipe's total_tokens according to component weights.
    """
    if not recipe.components or recipe.total_tokens == 0:
        return TokenBudget(total_tokens=recipe.total_tokens)

    normalized = recipe.normalized_weights
    profile_map = {p.name: p for p in profiles}

    allocations: Dict[str, int] = {}
    overflow = 0

    for name, weight in normalized.items():
        target = int(recipe.total_tokens * weight)
        available = profile_map.get(name, DatasetProfile(name=name)).n_tokens

        if available > 0 and target > available:
            allocations[name] = available
            overflow += target - available
        else:
            allocations[name] = target

    total_allocated = sum(allocations.values())
    utilization = total_allocated / recipe.total_tokens if recipe.total_tokens > 0 else 0.0

    return TokenBudget(
        total_tokens=recipe.total_tokens,
        allocations=allocations,
        overflow=overflow,
        utilization=utilization,
    )


def fit_to_budget(
    profiles: List[DatasetProfile],
    token_budget: int,
    strategy: str = "proportional",
) -> TokenBudget:
    """Fit datasets to a fixed token budget.

    Args:
        profiles: Dataset profiles to fit.
        token_budget: Total token budget.
        strategy: 'proportional' (by size) or 'equal'.
    """
    if not profiles or token_budget <= 0:
        return TokenBudget(total_tokens=token_budget)

    allocations: Dict[str, int] = {}

    if strategy == "equal":
        per_dataset = token_budget // len(profiles)
        for p in profiles:
            allocations[p.name] = min(per_dataset, p.n_tokens) if p.n_tokens > 0 else per_dataset
    else:
        # Proportional
        total_available = sum(p.n_tokens for p in profiles)
        if total_available == 0:
            per_dataset = token_budget // len(profiles)
            for p in profiles:
                allocations[p.name] = per_dataset
        else:
            for p in profiles:
                frac = p.n_tokens / total_available
                allocations[p.name] = int(token_budget * frac)

    total_allocated = sum(allocations.values())
    overflow = max(0, token_budget - total_allocated)
    utilization = total_allocated / token_budget if token_budget > 0 else 0.0

    return TokenBudget(
        total_tokens=token_budget,
        allocations=allocations,
        overflow=overflow,
        utilization=utilization,
    )


def budget_report(budget: TokenBudget) -> str:
    """Format a budget as a text report."""
    lines: List[str] = []
    lines.append("Token Budget Report")
    lines.append("=" * 50)
    lines.append(f"  Total budget:  {budget.total_tokens:>15,} tokens")
    lines.append(f"  Allocated:     {sum(budget.allocations.values()):>15,} tokens")
    lines.append(f"  Overflow:      {budget.overflow:>15,} tokens")
    lines.append(f"  Utilization:   {budget.utilization:>14.1%}")
    lines.append("")
    lines.append("  Allocations:")
    for name, tokens in sorted(budget.allocations.items(), key=lambda x: -x[1]):
        pct = tokens / budget.total_tokens * 100 if budget.total_tokens > 0 else 0
        lines.append(f"    {name:20s} {tokens:>12,} tokens ({pct:.1f}%)")
    return "\n".join(lines)
