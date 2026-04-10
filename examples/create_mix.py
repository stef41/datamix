"""Create, scale, and merge dataset mix recipes.

Demonstrates: create_recipe(), scale_recipe(), merge_recipes().
"""

from datamix import (
    DatasetProfile,
    MixStrategy,
    QualityMetrics,
    create_recipe,
    merge_recipes,
    scale_recipe,
)

if __name__ == "__main__":
    # Define dataset profiles (typically from profiling real data)
    profiles = [
        DatasetProfile(
            name="openassistant",
            n_examples=35_000,
            n_tokens=12_000_000,
            avg_tokens_per_example=342.9,
            quality=QualityMetrics(avg_quality_score=0.82),
            categories={"chat": 20000, "coding": 10000, "math": 5000},
        ),
        DatasetProfile(
            name="dolly-15k",
            n_examples=15_000,
            n_tokens=4_500_000,
            avg_tokens_per_example=300.0,
            quality=QualityMetrics(avg_quality_score=0.75),
            categories={"qa": 8000, "summary": 4000, "creative": 3000},
        ),
        DatasetProfile(
            name="code-alpaca",
            n_examples=20_000,
            n_tokens=8_000_000,
            avg_tokens_per_example=400.0,
            quality=QualityMetrics(avg_quality_score=0.88),
            categories={"coding": 18000, "explanation": 2000},
        ),
    ]

    # Create a proportional recipe (weight by token count)
    recipe = create_recipe(profiles, strategy=MixStrategy.PROPORTIONAL)
    print("=== Proportional Recipe ===")
    for name, weight in sorted(recipe.components.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {weight:.2%}")
    print(f"  Total tokens: {recipe.total_tokens:,}")

    # Create a temperature-scaled recipe (more uniform distribution)
    temp_recipe = create_recipe(profiles, strategy=MixStrategy.TEMPERATURE, temperature=2.0)
    print("\n=== Temperature-Scaled Recipe (T=2.0) ===")
    for name, weight in sorted(temp_recipe.components.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {weight:.2%}")

    # Scale a recipe to a target token budget
    scaled = scale_recipe(recipe, target_tokens=5_000_000)
    print(f"\n=== Scaled to 5M tokens ===")
    print(f"  Total tokens: {scaled.total_tokens:,}")
    for name, weight in sorted(scaled.components.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {weight:.2%}")

    # Merge two recipes with custom weights
    merged = merge_recipes([recipe, temp_recipe], weights=[0.7, 0.3])
    print(f"\n=== Merged Recipe (70% proportional + 30% temperature) ===")
    for name, weight in sorted(merged.components.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {weight:.2%}")
