"""Optimize dataset mixing ratios automatically.

Demonstrates: optimize_ratios(), grid_search_ratios(), format_optimization().
"""

from datamix import (
    DatasetProfile,
    QualityMetrics,
    optimize_ratios,
    grid_search_ratios,
    format_optimization,
)

if __name__ == "__main__":
    # Define profiles for several data sources
    profiles = {
        "openassistant": DatasetProfile(
            name="openassistant",
            n_examples=35_000,
            n_tokens=12_000_000,
            quality=QualityMetrics(avg_quality_score=0.82),
            categories={"chat": 20000, "coding": 10000, "math": 5000},
        ),
        "dolly": DatasetProfile(
            name="dolly",
            n_examples=15_000,
            n_tokens=4_500_000,
            quality=QualityMetrics(avg_quality_score=0.75),
            categories={"qa": 8000, "summary": 4000, "creative": 3000},
        ),
        "code-alpaca": DatasetProfile(
            name="code-alpaca",
            n_examples=20_000,
            n_tokens=8_000_000,
            quality=QualityMetrics(avg_quality_score=0.88),
            categories={"coding": 18000, "explanation": 2000},
        ),
    }

    # --- 1. Iterative optimization for "balanced" objective ---
    result = optimize_ratios(profiles, objective="balanced")
    print("=== Iterative Optimization (balanced) ===")
    print(format_optimization(result))

    # --- 2. Optimize for diversity ---
    result_div = optimize_ratios(profiles, objective="diversity")
    print("\n=== Iterative Optimization (diversity) ===")
    print(format_optimization(result_div))

    # --- 3. Grid search: try all combos at 10-step granularity ---
    grid_results = grid_search_ratios(profiles, objective="quality", steps=10)
    print(f"\n=== Grid Search (quality, {len(grid_results)} combos) ===")
    print("Top 5 configurations:")
    for i, res in enumerate(grid_results[:5], 1):
        weights_str = ", ".join(f"{k}={v:.0%}" for k, v in sorted(res.weights.items()))
        print(f"  {i}. score={res.score:.4f}  [{weights_str}]")

    print(f"\nWorst configuration:")
    worst = grid_results[-1]
    weights_str = ", ".join(f"{k}={v:.0%}" for k, v in sorted(worst.weights.items()))
    print(f"     score={worst.score:.4f}  [{weights_str}]")
