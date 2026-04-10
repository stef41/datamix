"""Tests for datamix.calculator."""

import pytest
from datamix.calculator import (
    BudgetCalculator,
    CalcTokenBudget,
    DatasetCost,
    estimate_tokens,
    format_budget_report,
)


@pytest.fixture
def calc():
    c = BudgetCalculator(budget_tokens=1_000_000, cost_per_1k_tokens=0.01)
    c.add_dataset("alpaca", 500_000)
    c.add_dataset("dolly", 300_000)
    c.add_dataset("oasst", 200_000)
    return c


class TestCalcTokenBudget:
    def test_fields(self):
        b = CalcTokenBudget(
            total_tokens=1000,
            cost_per_token=0.00001,
            total_cost=0.01,
            breakdown={"a": 1000},
        )
        assert b.total_tokens == 1000
        assert b.total_cost == 0.01


class TestDatasetCost:
    def test_fields(self):
        dc = DatasetCost(name="test", tokens=1000, proportion=0.5, cost=0.01)
        assert dc.name == "test"
        assert dc.proportion == 0.5


class TestBudgetCalculatorInit:
    def test_budget_from_tokens(self):
        c = BudgetCalculator(budget_tokens=100_000)
        assert c._budget_tokens == 100_000

    def test_budget_from_dollars(self):
        c = BudgetCalculator(budget_dollars=10.0, cost_per_1k_tokens=0.01)
        # $10 / ($0.01/1000) = 1_000_000 (int truncation may lose 1)
        assert c._budget_tokens == pytest.approx(1_000_000, abs=1)

    def test_invalid_cost(self):
        with pytest.raises(ValueError, match="cost_per_1k_tokens"):
            BudgetCalculator(cost_per_1k_tokens=0)

    def test_negative_budget_tokens(self):
        with pytest.raises(ValueError, match="budget_tokens"):
            BudgetCalculator(budget_tokens=-1)

    def test_negative_budget_dollars(self):
        with pytest.raises(ValueError, match="budget_dollars"):
            BudgetCalculator(budget_dollars=-1.0)


class TestAddDataset:
    def test_basic(self):
        c = BudgetCalculator()
        c.add_dataset("test", 10_000)
        assert c._datasets["test"] == 10_000

    def test_with_avg_tokens(self):
        c = BudgetCalculator()
        c.add_dataset("test", 100, avg_tokens_per_example=50)
        assert c._datasets["test"] == 5000

    def test_empty_name_raises(self):
        c = BudgetCalculator()
        with pytest.raises(ValueError, match="name"):
            c.add_dataset("", 100)

    def test_negative_tokens_raises(self):
        c = BudgetCalculator()
        with pytest.raises(ValueError, match="tokens_or_examples"):
            c.add_dataset("test", -1)

    def test_invalid_avg_tokens(self):
        c = BudgetCalculator()
        with pytest.raises(ValueError, match="avg_tokens_per_example"):
            c.add_dataset("test", 100, avg_tokens_per_example=0)


class TestCalculate:
    def test_total_tokens(self, calc):
        result = calc.calculate()
        assert result.total_tokens == 1_000_000

    def test_cost(self, calc):
        result = calc.calculate()
        # 1_000_000 tokens * ($0.01 / 1000) = $10.00
        assert result.total_cost == pytest.approx(10.0)

    def test_breakdown_keys(self, calc):
        result = calc.calculate()
        assert set(result.breakdown.keys()) == {"alpaca", "dolly", "oasst"}

    def test_empty_calculator(self):
        c = BudgetCalculator()
        result = c.calculate()
        assert result.total_tokens == 0
        assert result.total_cost == 0.0


class TestOptimize:
    def test_proportional_scaling(self, calc):
        # Current total is 1M, scale to 500K
        opt = calc.optimize(500_000)
        assert sum(opt.values()) <= 500_000
        assert opt["alpaca"] > opt["dolly"] > opt["oasst"]

    def test_zero_target(self, calc):
        opt = calc.optimize(0)
        assert all(v == 0 for v in opt.values())

    def test_negative_target_raises(self, calc):
        with pytest.raises(ValueError, match="target_budget"):
            calc.optimize(-1)

    def test_empty_datasets(self):
        c = BudgetCalculator()
        assert c.optimize(100_000) == {}


class TestCostBreakdown:
    def test_three_datasets(self, calc):
        costs = calc.cost_breakdown()
        assert len(costs) == 3

    def test_proportions_sum_to_one(self, calc):
        costs = calc.cost_breakdown()
        total = sum(c.proportion for c in costs)
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_cost_values_positive(self, calc):
        for c in calc.cost_breakdown():
            assert c.cost > 0


class TestRemainingBudget:
    def test_partial_use(self, calc):
        # Budget = 1M, used = 1M → 0 remaining
        assert calc.remaining_budget() == 0

    def test_under_budget(self):
        c = BudgetCalculator(budget_tokens=2_000_000)
        c.add_dataset("x", 500_000)
        assert c.remaining_budget() == 1_500_000

    def test_no_budget_set(self):
        c = BudgetCalculator()
        c.add_dataset("x", 100)
        assert c.remaining_budget() == 0


class TestScaleToBudget:
    def test_double(self, calc):
        scaled = calc.scale_to_budget(2_000_000)
        assert scaled["alpaca"] == 1_000_000

    def test_halve(self, calc):
        scaled = calc.scale_to_budget(500_000)
        assert scaled["alpaca"] == 250_000

    def test_negative_raises(self, calc):
        with pytest.raises(ValueError, match="target_tokens"):
            calc.scale_to_budget(-1)


class TestEstimateTokens:
    def test_simple(self):
        assert estimate_tokens("hello world") == int(2 * 1.3)

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestFormatBudgetReport:
    def test_contains_totals(self, calc):
        budget = calc.calculate()
        report = format_budget_report(budget)
        assert "Token Budget Report" in report
        assert "1,000,000" in report

    def test_contains_datasets(self, calc):
        budget = calc.calculate()
        report = format_budget_report(budget)
        assert "alpaca" in report
        assert "dolly" in report
