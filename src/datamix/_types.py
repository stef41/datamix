"""Core types for datamix."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DatamixError(Exception):
    """Base exception for datamix."""


class MixStrategy(Enum):
    """How to combine datasets."""
    PROPORTIONAL = "proportional"
    TEMPERATURE = "temperature"
    EQUAL = "equal"
    CUSTOM = "custom"


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset or subset."""
    avg_length: float = 0.0
    median_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    std_length: float = 0.0
    n_duplicates: int = 0
    duplicate_rate: float = 0.0
    avg_quality_score: float = 0.0
    language_distribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """Profile of a single dataset."""
    name: str
    n_examples: int = 0
    n_tokens: int = 0
    avg_tokens_per_example: float = 0.0
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    categories: Dict[str, int] = field(default_factory=dict)

    @property
    def size_tokens_m(self) -> float:
        return self.n_tokens / 1_000_000 if self.n_tokens > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_examples": self.n_examples,
            "n_tokens": self.n_tokens,
            "avg_tokens_per_example": round(self.avg_tokens_per_example, 1),
            "size_tokens_m": round(self.size_tokens_m, 2),
            "categories": self.categories,
            "quality": {
                "avg_length": round(self.quality.avg_length, 1),
                "median_length": round(self.quality.median_length, 1),
                "min_length": self.quality.min_length,
                "max_length": self.quality.max_length,
                "n_duplicates": self.quality.n_duplicates,
                "duplicate_rate": round(self.quality.duplicate_rate, 4),
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DatasetProfile:
        qd = d.get("quality", {})
        quality = QualityMetrics(
            avg_length=qd.get("avg_length", 0),
            median_length=qd.get("median_length", 0),
            min_length=qd.get("min_length", 0),
            max_length=qd.get("max_length", 0),
            n_duplicates=qd.get("n_duplicates", 0),
            duplicate_rate=qd.get("duplicate_rate", 0),
        )
        return cls(
            name=d["name"],
            n_examples=d.get("n_examples", 0),
            n_tokens=d.get("n_tokens", 0),
            avg_tokens_per_example=d.get("avg_tokens_per_example", 0),
            quality=quality,
            categories=d.get("categories", {}),
            metadata=d.get("metadata", {}),
        )


@dataclass
class MixRecipe:
    """A recipe for mixing datasets with specific weights."""
    name: str = "mix"
    components: Dict[str, float] = field(default_factory=dict)  # dataset_name → weight
    strategy: MixStrategy = MixStrategy.PROPORTIONAL
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def normalized_weights(self) -> Dict[str, float]:
        total = sum(self.components.values())
        if total == 0:
            return {}
        return {k: round(v / total, 6) for k, v in self.components.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "components": self.components,
            "normalized_weights": self.normalized_weights,
            "strategy": self.strategy.value,
            "total_tokens": self.total_tokens,
            "n_components": self.n_components,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MixRecipe:
        return cls(
            name=d.get("name", "mix"),
            components=d.get("components", {}),
            strategy=MixStrategy(d.get("strategy", "proportional")),
            total_tokens=d.get("total_tokens", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class CurriculumPhase:
    """A single phase in a curriculum schedule."""
    name: str
    start_fraction: float  # 0-1 progress through training
    end_fraction: float
    weights: Dict[str, float]  # dataset_name → weight for this phase

    @property
    def duration(self) -> float:
        return self.end_fraction - self.start_fraction


@dataclass
class CurriculumSchedule:
    """A multi-phase training curriculum."""
    name: str = "curriculum"
    phases: List[CurriculumPhase] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def weights_at(self, progress: float) -> Dict[str, float]:
        """Get dataset weights at a given training progress (0-1)."""
        progress = max(0.0, min(1.0, progress))
        for phase in self.phases:
            if phase.start_fraction <= progress < phase.end_fraction:
                return phase.weights
        # Return last phase weights if at end
        if self.phases:
            return self.phases[-1].weights
        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_phases": self.n_phases,
            "total_tokens": self.total_tokens,
            "phases": [
                {
                    "name": p.name,
                    "start": p.start_fraction,
                    "end": p.end_fraction,
                    "duration": round(p.duration, 4),
                    "weights": p.weights,
                }
                for p in self.phases
            ],
        }


@dataclass
class SamplingConfig:
    """Configuration for dataset sampling."""
    seed: int = 42
    temperature: float = 1.0
    max_examples: Optional[int] = None
    stratify_by: Optional[str] = None
    deduplicate: bool = False


@dataclass
class TokenBudget:
    """Token budget allocation across datasets."""
    total_tokens: int = 0
    allocations: Dict[str, int] = field(default_factory=dict)  # dataset → tokens
    overflow: int = 0  # tokens that couldn't be allocated
    utilization: float = 0.0  # fraction of budget actually used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "allocations": self.allocations,
            "overflow": self.overflow,
            "utilization": round(self.utilization, 4),
            "n_datasets": len(self.allocations),
        }
