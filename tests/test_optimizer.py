"""Tests for datamix.optimizer — automatic dataset ratio optimization."""

from __future__ import annotations

import pytest

from datamix._types import DatamixError, DatasetProfile, QualityMetrics
from datamix.optimizer import (
    OptimizationResult,
    format_optimization,
    grid_search_ratios,
    optimize_ratios,
)


# ── helpers ────────────────────────────────────────────────────────────


def _prof(
    name: str,
    n_tokens: int = 1_000_000,
    quality: float = 0.8,
    categories: dict[str, int] | None = None,
) -> DatasetProfile:
    return DatasetProfile(
        name=name,
        n_examples=n_tokens // 100,
        n_tokens=n_tokens,
        avg_tokens_per_example=100.0,
        quality=QualityMetrics(avg_quality_score=quality),
        categories=categories or {},
    )


def _base_profiles() -> dict[str, DatasetProfile]:
    return {
        "wiki": _prof("wiki", 5_000_000, 0.9, {"knowledge": 100}),
        "code": _prof("code", 3_000_000, 0.85, {"code": 80}),
        "chat": _prof("chat", 2_000_000, 0.7, {"dialogue": 60}),
    }


# ── OptimizationResult dataclass ──────────────────────────────────────


class TestOptimizationResult:
    def test_fields(self):
        r = OptimizationResult(weights={"a": 0.5, "b": 0.5}, score=0.9, iterations=10, converged=True)
        assert r.weights == {"a": 0.5, "b": 0.5}
        assert r.score == 0.9
        assert r.iterations == 10
        assert r.converged is True


# ── optimize_ratios ───────────────────────────────────────────────────


class TestOptimizeRatios:
    def test_empty_profiles(self):
        r = optimize_ratios({})
        assert r.weights == {}
        assert r.converged is True

    def test_single_source(self):
        profiles = {"only": _prof("only")}
        r = optimize_ratios(profiles, objective="balanced")
        assert len(r.weights) == 1
        assert pytest.approx(r.weights["only"], abs=1e-4) == 1.0

    def test_balanced_returns_result(self):
        r = optimize_ratios(_base_profiles(), objective="balanced")
        assert isinstance(r, OptimizationResult)
        total = sum(r.weights.values())
        assert pytest.approx(total, abs=1e-3) == 1.0

    def test_quality_favours_high_quality(self):
        profiles = {
            "good": _prof("good", quality=0.95),
            "bad": _prof("bad", quality=0.1),
        }
        r = optimize_ratios(profiles, objective="quality")
        assert r.weights["good"] > r.weights["bad"]

    def test_diversity_objective(self):
        profiles = {
            "a": _prof("a", categories={"cat1": 50}),
            "b": _prof("b", categories={"cat2": 50}),
        }
        r = optimize_ratios(profiles, objective="diversity")
        # Both sources contribute unique cats, expect roughly equal
        assert abs(r.weights["a"] - r.weights["b"]) < 0.3

    def test_invalid_objective_raises(self):
        with pytest.raises(DatamixError, match="Unknown objective"):
            optimize_ratios(_base_profiles(), objective="bogus")

    def test_weights_sum_to_one(self):
        r = optimize_ratios(_base_profiles(), objective="quality")
        assert pytest.approx(sum(r.weights.values()), abs=1e-3) == 1.0

    def test_token_budget_constraint(self):
        profiles = {
            "small": _prof("small", n_tokens=100),
            "big": _prof("big", n_tokens=10_000_000),
        }
        r = optimize_ratios(profiles, token_budget=200, objective="balanced")
        # small can only contribute 100 tokens out of 200 budget → at most 50 %
        assert r.weights["small"] <= 0.55

    def test_converges(self):
        r = optimize_ratios(_base_profiles(), objective="balanced", max_iterations=500)
        assert r.converged is True

    def test_score_non_negative(self):
        r = optimize_ratios(_base_profiles(), objective="balanced")
        assert r.score >= 0.0


# ── grid_search_ratios ────────────────────────────────────────────────


class TestGridSearchRatios:
    def test_empty_profiles(self):
        assert grid_search_ratios({}) == []

    def test_single_source(self):
        results = grid_search_ratios({"only": _prof("only")}, steps=5)
        assert len(results) >= 1
        assert results[0].weights["only"] == pytest.approx(1.0, abs=1e-4)

    def test_two_sources_steps(self):
        profiles = {"a": _prof("a"), "b": _prof("b")}
        results = grid_search_ratios(profiles, steps=5)
        # 2 sources, 5 steps ⇒ C(6,1) = 6 combos
        assert len(results) == 6

    def test_sorted_by_score(self):
        results = grid_search_ratios(_base_profiles(), steps=4)
        for a, b in zip(results, results[1:]):
            assert a.score >= b.score

    def test_invalid_objective(self):
        with pytest.raises(DatamixError):
            grid_search_ratios(_base_profiles(), objective="nope")

    def test_all_weights_sum_to_one(self):
        results = grid_search_ratios(_base_profiles(), steps=3)
        for r in results:
            assert pytest.approx(sum(r.weights.values()), abs=1e-3) == 1.0


# ── format_optimization ──────────────────────────────────────────────


class TestFormatOptimization:
    def test_contains_key_fields(self):
        r = OptimizationResult(
            weights={"wiki": 0.5, "code": 0.3, "chat": 0.2},
            score=0.87,
            iterations=42,
            converged=True,
        )
        text = format_optimization(r)
        assert "0.87" in text
        assert "True" in text
        assert "42" in text
        assert "wiki" in text

    def test_empty_weights(self):
        r = OptimizationResult(weights={}, score=0.0, iterations=0, converged=True)
        text = format_optimization(r)
        assert "Weights:" in text
