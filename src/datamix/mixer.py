"""Mix recipes — create, merge, and scale dataset blend configurations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from datamix._types import DatasetProfile, MixRecipe, MixStrategy


def create_recipe(
    profiles: List[DatasetProfile],
    strategy: MixStrategy = MixStrategy.PROPORTIONAL,
    weights: Optional[Dict[str, float]] = None,
    total_tokens: int = 0,
    temperature: float = 1.0,
) -> MixRecipe:
    """Create a mix recipe from dataset profiles.

    Args:
        profiles: List of dataset profiles to mix.
        strategy: Mixing strategy.
        weights: Custom weights (only for CUSTOM strategy).
        total_tokens: Target total tokens for the mix.
        temperature: Temperature for TEMPERATURE strategy (>1 = more uniform).
    """
    if not profiles:
        return MixRecipe()

    components: Dict[str, float] = {}

    if strategy == MixStrategy.CUSTOM and weights:
        components = dict(weights)
    elif strategy == MixStrategy.EQUAL:
        w = 1.0 / len(profiles)
        for p in profiles:
            components[p.name] = w
    elif strategy == MixStrategy.TEMPERATURE:
        # Temperature-scaled proportional: w_i = (n_i ^ (1/T)) / sum(n_j ^ (1/T))
        raw: Dict[str, float] = {}
        for p in profiles:
            n = max(1, p.n_tokens)
            raw[p.name] = n ** (1.0 / max(0.01, temperature))
        total = sum(raw.values())
        components = {k: v / total for k, v in raw.items()}
    else:
        # Proportional: weight by token count
        total = sum(max(1, p.n_tokens) for p in profiles)
        for p in profiles:
            components[p.name] = max(1, p.n_tokens) / total

    if total_tokens == 0:
        total_tokens = sum(p.n_tokens for p in profiles)

    return MixRecipe(
        name=f"mix_{strategy.value}",
        components=components,
        strategy=strategy,
        total_tokens=total_tokens,
    )


def merge_recipes(recipes: List[MixRecipe], weights: Optional[List[float]] = None) -> MixRecipe:
    """Merge multiple recipes into one, optionally weighted.

    Args:
        recipes: List of mix recipes.
        weights: Optional weights for each recipe (default: equal).
    """
    if not recipes:
        return MixRecipe()

    if weights is None:
        weights = [1.0 / len(recipes)] * len(recipes)

    if len(weights) != len(recipes):
        raise ValueError("weights must have same length as recipes")

    total_w = sum(weights)
    if total_w == 0:
        return MixRecipe()

    merged: Dict[str, float] = {}
    for recipe, w in zip(recipes, weights):
        scale = w / total_w
        for name, comp_w in recipe.components.items():
            merged[name] = merged.get(name, 0.0) + comp_w * scale

    total_tokens = sum(r.total_tokens for r in recipes)

    return MixRecipe(
        name="merged",
        components=merged,
        strategy=MixStrategy.CUSTOM,
        total_tokens=total_tokens,
    )


def scale_recipe(recipe: MixRecipe, factor: float) -> MixRecipe:
    """Scale a recipe's total tokens by a factor."""
    return MixRecipe(
        name=recipe.name,
        components=dict(recipe.components),
        strategy=recipe.strategy,
        total_tokens=int(recipe.total_tokens * factor),
        metadata=dict(recipe.metadata),
    )
