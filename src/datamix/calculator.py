"""Token budget calculator with cost awareness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CalcTokenBudget:
    """Result of a budget calculation."""

    total_tokens: int
    cost_per_token: float
    total_cost: float
    breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetCost:
    """Cost breakdown for a single dataset."""

    name: str
    tokens: int
    proportion: float
    cost: float


class BudgetCalculator:
    """Calculate and optimise token budgets with cost tracking.

    Provide *either* ``budget_tokens`` or ``budget_dollars`` (or both).
    ``cost_per_1k_tokens`` is the unit price used for all cost conversions.
    """

    def __init__(
        self,
        budget_tokens: Optional[int] = None,
        budget_dollars: Optional[float] = None,
        cost_per_1k_tokens: float = 0.01,
    ) -> None:
        if cost_per_1k_tokens <= 0:
            raise ValueError("cost_per_1k_tokens must be positive")

        self.cost_per_1k_tokens = cost_per_1k_tokens
        self._cost_per_token = cost_per_1k_tokens / 1000.0
        self._datasets: Dict[str, int] = {}  # name → tokens

        # Resolve the authoritative budget
        if budget_tokens is not None and budget_tokens < 0:
            raise ValueError("budget_tokens must be non-negative")
        if budget_dollars is not None and budget_dollars < 0:
            raise ValueError("budget_dollars must be non-negative")

        if budget_tokens is not None:
            self._budget_tokens = budget_tokens
        elif budget_dollars is not None:
            self._budget_tokens = int(budget_dollars / self._cost_per_token)
        else:
            self._budget_tokens = 0  # no cap

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_dataset(
        self,
        name: str,
        tokens_or_examples: int,
        avg_tokens_per_example: Optional[int] = None,
    ) -> None:
        """Register a dataset.

        If *avg_tokens_per_example* is provided, ``tokens_or_examples`` is
        treated as an example count and multiplied accordingly.
        """
        if not name:
            raise ValueError("name must be a non-empty string")
        if tokens_or_examples < 0:
            raise ValueError("tokens_or_examples must be non-negative")

        if avg_tokens_per_example is not None:
            if avg_tokens_per_example <= 0:
                raise ValueError("avg_tokens_per_example must be positive")
            tokens = tokens_or_examples * avg_tokens_per_example
        else:
            tokens = tokens_or_examples

        self._datasets[name] = tokens

    def calculate(self) -> CalcTokenBudget:
        """Calculate the total budget and per-dataset breakdown."""
        total = sum(self._datasets.values())
        return CalcTokenBudget(
            total_tokens=total,
            cost_per_token=self._cost_per_token,
            total_cost=round(total * self._cost_per_token, 6),
            breakdown=dict(self._datasets),
        )

    def optimize(self, target_budget: int) -> Dict[str, int]:
        """Proportionally scale datasets to fit *target_budget* tokens."""
        if target_budget < 0:
            raise ValueError("target_budget must be non-negative")
        total = sum(self._datasets.values())
        if total == 0:
            return {name: 0 for name in self._datasets}

        result: Dict[str, int] = {}
        for name, tokens in self._datasets.items():
            result[name] = int(tokens / total * target_budget)
        return result

    def cost_breakdown(self) -> List[DatasetCost]:
        """Return per-dataset cost information."""
        total = sum(self._datasets.values())
        results: List[DatasetCost] = []
        for name, tokens in self._datasets.items():
            proportion = tokens / total if total > 0 else 0.0
            cost = round(tokens * self._cost_per_token, 6)
            results.append(
                DatasetCost(
                    name=name,
                    tokens=tokens,
                    proportion=round(proportion, 6),
                    cost=cost,
                )
            )
        return results

    def remaining_budget(self) -> int:
        """Tokens remaining before hitting the budget cap.

        Returns 0 when no budget cap was set.
        """
        if self._budget_tokens == 0:
            return 0
        used = sum(self._datasets.values())
        return max(0, self._budget_tokens - used)

    def scale_to_budget(self, target_tokens: int) -> Dict[str, int]:
        """Scale all datasets proportionally to reach *target_tokens*."""
        if target_tokens < 0:
            raise ValueError("target_tokens must be non-negative")
        total = sum(self._datasets.values())
        if total == 0:
            return {name: 0 for name in self._datasets}

        scale = target_tokens / total
        return {name: int(tokens * scale) for name, tokens in self._datasets.items()}


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ``words × 1.3``."""
    words = len(text.split())
    return int(words * 1.3)


def format_budget_report(budget: CalcTokenBudget) -> str:
    """Human-readable budget report."""
    lines: List[str] = []
    lines.append("Token Budget Report")
    lines.append("=" * 50)
    lines.append(f"Total tokens : {budget.total_tokens:,}")
    lines.append(f"Cost/token   : ${budget.cost_per_token:.6f}")
    lines.append(f"Total cost   : ${budget.total_cost:,.2f}")
    lines.append("")

    if budget.breakdown:
        lines.append(f"{'Dataset':<25} {'Tokens':>15}")
        lines.append("-" * 50)
        for name, tokens in sorted(
            budget.breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"{name:<25} {tokens:>15,}")

    lines.append("")
    return "\n".join(lines)
