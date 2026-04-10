"""datamix — Dataset mixing & curriculum optimizer."""

from __future__ import annotations

from datamix._types import (
    CurriculumPhase,
    CurriculumSchedule,
    DatamixError,
    DatasetProfile,
    MixRecipe,
    MixStrategy,
    QualityMetrics,
    SamplingConfig,
    TokenBudget,
)
from datamix.profile import (
    profile_dataset,
    profile_jsonl,
    compare_profiles,
)
from datamix.mixer import (
    create_recipe,
    merge_recipes,
    scale_recipe,
)
from datamix.curriculum import (
    linear_schedule,
    cosine_schedule,
    step_schedule,
    custom_schedule,
)
from datamix.sampler import (
    temperature_sample,
    proportional_sample,
    stratified_sample,
)
from datamix.budget import (
    compute_budget,
    fit_to_budget,
    budget_report,
)
from datamix.quality import (
    length_filter,
    dedup_exact,
    dedup_ngram,
    quality_score,
)

__all__ = [
    "CurriculumPhase",
    "CurriculumSchedule",
    "DatamixError",
    "DatasetProfile",
    "MixRecipe",
    "MixStrategy",
    "QualityMetrics",
    "SamplingConfig",
    "TokenBudget",
    "budget_report",
    "compare_profiles",
    "compute_budget",
    "cosine_schedule",
    "create_recipe",
    "custom_schedule",
    "dedup_exact",
    "dedup_ngram",
    "fit_to_budget",
    "length_filter",
    "linear_schedule",
    "merge_recipes",
    "profile_dataset",
    "profile_jsonl",
    "proportional_sample",
    "quality_score",
    "scale_recipe",
    "step_schedule",
    "stratified_sample",
    "temperature_sample",
]
