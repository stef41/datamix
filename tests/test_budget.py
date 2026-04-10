"""Tests for datamix.budget."""

import pytest
from datamix._types import DatasetProfile, MixRecipe, TokenBudget
from datamix.budget import budget_report, compute_budget, fit_to_budget


def _profiles():
    return [
        DatasetProfile(name="wiki", n_tokens=50000),
        DatasetProfile(name="code", n_tokens=30000),
        DatasetProfile(name="books", n_tokens=20000),
    ]


class TestComputeBudget:
    def test_basic(self):
        recipe = MixRecipe(
            components={"wiki": 0.5, "code": 0.3, "books": 0.2},
            total_tokens=100000,
        )
        budget = compute_budget(recipe, _profiles())
        assert budget.total_tokens == 100000
        assert len(budget.allocations) == 3
        assert budget.allocations["wiki"] > budget.allocations["code"]

    def test_overflow_capped(self):
        recipe = MixRecipe(
            components={"wiki": 0.9, "code": 0.1},
            total_tokens=1000000,  # more than available
        )
        budget = compute_budget(recipe, _profiles())
        # wiki has only 50000 tokens, so allocation should be capped
        assert budget.allocations["wiki"] <= 50000

    def test_empty(self):
        recipe = MixRecipe()
        budget = compute_budget(recipe, [])
        assert budget.total_tokens == 0


class TestFitToBudget:
    def test_proportional(self):
        budget = fit_to_budget(_profiles(), token_budget=50000)
        assert budget.total_tokens == 50000
        assert budget.allocations["wiki"] > budget.allocations["books"]

    def test_equal(self):
        budget = fit_to_budget(_profiles(), token_budget=30000, strategy="equal")
        # Equal: 10000 each, but books only has 20000 so capped
        for name, tokens in budget.allocations.items():
            assert tokens <= 10000

    def test_empty(self):
        budget = fit_to_budget([], token_budget=1000)
        assert budget.total_tokens == 1000
        assert len(budget.allocations) == 0


class TestBudgetReport:
    def test_basic(self):
        budget = TokenBudget(
            total_tokens=100000,
            allocations={"wiki": 50000, "code": 30000, "books": 20000},
            utilization=1.0,
        )
        text = budget_report(budget)
        assert "100,000" in text
        assert "wiki" in text
        assert "Utilization" in text
