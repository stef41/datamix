"""Edge-case and hardened tests for datamix modules."""

import pytest
from datamix._types import (
    CurriculumPhase,
    CurriculumSchedule,
    DatasetProfile,
    MixRecipe,
    TokenBudget,
)
from datamix.curriculum import cosine_schedule, custom_schedule, linear_schedule, step_schedule
from datamix.mixer import create_recipe, merge_recipes, scale_recipe
from datamix._types import MixStrategy
from datamix.quality import dedup_exact, dedup_ngram, length_filter, quality_score
from datamix.sampler import proportional_sample, stratified_sample, temperature_sample
from datamix.budget import budget_report, compute_budget, fit_to_budget


# ---- quality edge cases ----


class TestQualityEdgeCases:
    def test_quality_score_only_punctuation(self):
        score = quality_score("!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()")
        assert score < 0.5

    def test_quality_score_all_spaces(self):
        score = quality_score("              " * 10)
        # All spaces -> low alnum ratio, no words
        assert score < 0.5

    def test_quality_score_very_long_good(self):
        text = "This is a well-written document about machine learning. " * 20
        score = quality_score(text)
        assert score > 0.5

    def test_quality_score_no_punctuation_threshold(self):
        # 11 words, no punctuation => 0.8 penalty
        text = "one two three four five six seven eight nine ten eleven"
        score = quality_score(text, min_length=10)
        assert score <= 0.8

    def test_quality_score_high_repetition(self):
        text = "spam " * 50
        score = quality_score(text, min_length=10)
        assert score < 0.5

    def test_dedup_exact_unicode(self):
        examples = ["日本語", "日本語", "中文"]
        kept, stats = dedup_exact(examples)
        assert stats["kept"] == 2
        assert stats["duplicates_removed"] == 1

    def test_dedup_exact_empty_strings(self):
        examples = ["", "", "hello"]
        kept, stats = dedup_exact(examples)
        assert stats["kept"] == 2  # "" and "hello"

    def test_dedup_ngram_short_texts(self):
        examples = ["hi", "hello", "hey"]
        kept, stats = dedup_ngram(examples, n=5, threshold=0.8)
        # Short texts have tiny ngram sets, should not crash
        assert isinstance(kept, list)

    def test_dedup_ngram_identical(self):
        text = "the quick brown fox jumps over the lazy dog"
        examples = [text, text, text]
        kept, stats = dedup_ngram(examples, n=3, threshold=0.5)
        assert stats["kept"] == 1

    def test_length_filter_combined(self):
        examples = ["a", "hello world foo bar baz", "x" * 1000]
        kept, stats = length_filter(examples, min_length=5, max_length=100, min_words=2)
        assert stats["kept"] == 1  # only "hello world foo bar baz"

    def test_quality_score_max_repetition_param(self):
        text = "good content with reasonable repetition patterns throughout. " * 5
        s1 = quality_score(text, max_repetition_ratio=0.1)
        s2 = quality_score(text, max_repetition_ratio=0.9)
        assert s2 >= s1


# ---- curriculum edge cases ----


class TestCurriculumEdgeCases:
    def test_linear_single_dataset(self):
        sched = linear_schedule(["wiki"], n_phases=3)
        for phase in sched.phases:
            assert phase.weights["wiki"] == pytest.approx(1.0)

    def test_linear_single_phase(self):
        sched = linear_schedule(["a", "b"], n_phases=1)
        assert sched.n_phases == 1
        assert sched.phases[0].start_fraction == 0.0
        assert sched.phases[0].end_fraction == 1.0

    def test_cosine_single_dataset(self):
        sched = cosine_schedule(["wiki"], n_phases=4)
        # With single dataset, it's always primary
        for phase in sched.phases:
            assert "wiki" in phase.weights

    def test_cosine_two_datasets_sums_to_one(self):
        sched = cosine_schedule(["a", "b"], n_phases=4)
        for phase in sched.phases:
            total = sum(phase.weights.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_step_schedule_empty(self):
        sched = step_schedule([])
        assert sched.n_phases == 0

    def test_step_schedule_single_phase(self):
        config = [{"name": "only", "fraction": 1.0, "weights": {"a": 1.0}}]
        sched = step_schedule(config)
        assert sched.n_phases == 1
        assert sched.phases[0].end_fraction == pytest.approx(1.0)

    def test_custom_schedule_preserves_phases(self):
        phases = [
            CurriculumPhase("p1", 0.0, 0.3, {"a": 0.7, "b": 0.3}),
            CurriculumPhase("p2", 0.3, 1.0, {"a": 0.3, "b": 0.7}),
        ]
        sched = custom_schedule(phases, name="custom", total_tokens=1000)
        assert sched.total_tokens == 1000
        assert sched.phases[0].weights["a"] == 0.7

    def test_linear_with_total_tokens(self):
        sched = linear_schedule(["a", "b"], n_phases=2, total_tokens=50000)
        assert sched.total_tokens == 50000


# ---- sampler edge cases ----


class TestSamplerEdgeCases:
    def test_temperature_sample_single_dataset(self):
        datasets = {"only": ["a", "b", "c"]}
        result = temperature_sample(datasets, n_samples=10)
        assert len(result) == 10
        assert all(r in ["a", "b", "c"] for r in result)

    def test_temperature_sample_temp_one_is_proportional(self):
        datasets = {
            "big": ["big"] * 100,
            "small": ["small"] * 10,
        }
        r1 = temperature_sample(datasets, n_samples=100, temperature=1.0, seed=42)
        r2 = proportional_sample(datasets, n_samples=100, seed=42)
        assert r1 == r2

    def test_stratified_empty_examples(self):
        result = stratified_sample([], lambda x: "a", n_samples=10)
        assert result == []

    def test_stratified_single_category(self):
        examples = ["item " + str(i) for i in range(20)]
        result = stratified_sample(examples, lambda x: "all", n_samples=5)
        assert len(result) == 5

    def test_stratified_deterministic(self):
        examples = ["cat " + str(i) for i in range(30)] + ["dog " + str(i) for i in range(30)]
        cat_fn = lambda x: x.split()[0]
        r1 = stratified_sample(examples, cat_fn, n_samples=20, seed=42)
        r2 = stratified_sample(examples, cat_fn, n_samples=20, seed=42)
        assert r1 == r2


# ---- mixer edge cases ----


class TestMixerEdgeCases:
    def test_create_recipe_proportional(self):
        profiles = [
            DatasetProfile(name="a", n_tokens=1000),
            DatasetProfile(name="b", n_tokens=500),
        ]
        recipe = create_recipe(profiles, strategy=MixStrategy.PROPORTIONAL, total_tokens=900)
        assert recipe.total_tokens == 900
        assert recipe.components["a"] > recipe.components["b"]

    def test_create_recipe_equal(self):
        profiles = [
            DatasetProfile(name="a", n_tokens=1000),
            DatasetProfile(name="b", n_tokens=500),
        ]
        recipe = create_recipe(profiles, strategy=MixStrategy.EQUAL, total_tokens=1000)
        assert recipe.components["a"] == pytest.approx(recipe.components["b"], abs=0.01)

    def test_create_recipe_temperature(self):
        profiles = [
            DatasetProfile(name="a", n_tokens=1000),
            DatasetProfile(name="b", n_tokens=100),
        ]
        recipe = create_recipe(profiles, strategy=MixStrategy.TEMPERATURE, total_tokens=500, temperature=5.0)
        # High temperature -> more uniform
        assert recipe.components["b"] > 0.2

    def test_scale_recipe(self):
        recipe = MixRecipe(
            components={"a": 0.6, "b": 0.4},
            total_tokens=1000,
        )
        scaled = scale_recipe(recipe, 2.0)
        assert scaled.total_tokens == 2000
        assert scaled.components == recipe.components

    def test_merge_recipes(self):
        r1 = MixRecipe(components={"a": 0.8, "b": 0.2}, total_tokens=1000)
        r2 = MixRecipe(components={"a": 0.3, "b": 0.7}, total_tokens=1000)
        merged = merge_recipes([r1, r2])
        assert "a" in merged.components
        assert "b" in merged.components

    def test_create_recipe_empty(self):
        recipe = create_recipe([])
        assert recipe.total_tokens == 0
        assert len(recipe.components) == 0


# ---- budget edge cases ----


class TestBudgetEdgeCases:
    def test_fit_to_budget_zero(self):
        profiles = [DatasetProfile(name="a", n_tokens=1000)]
        budget = fit_to_budget(profiles, token_budget=0)
        assert budget.total_tokens == 0

    def test_budget_report_empty(self):
        budget = TokenBudget(total_tokens=0, allocations={})
        text = budget_report(budget)
        assert isinstance(text, str)

    def test_compute_budget_missing_profile(self):
        recipe = MixRecipe(
            components={"a": 0.5, "b": 0.5},
            total_tokens=1000,
        )
        profiles = [DatasetProfile(name="a", n_tokens=5000)]
        # b has no profile -> should still compute
        budget = compute_budget(recipe, profiles)
        assert "a" in budget.allocations


# ---- types edge cases ----


class TestTypesEdgeCases:
    def test_mix_recipe_normalized_weights(self):
        recipe = MixRecipe(components={"a": 3.0, "b": 1.0})
        nw = recipe.normalized_weights
        assert nw["a"] == pytest.approx(0.75)
        assert nw["b"] == pytest.approx(0.25)

    def test_mix_recipe_empty(self):
        recipe = MixRecipe()
        nw = recipe.normalized_weights
        assert nw == {}

    def test_curriculum_schedule_weights_at(self):
        phases = [
            CurriculumPhase("p1", 0.0, 0.5, {"a": 0.8, "b": 0.2}),
            CurriculumPhase("p2", 0.5, 1.0, {"a": 0.3, "b": 0.7}),
        ]
        sched = CurriculumSchedule(name="test", phases=phases)
        w = sched.weights_at(0.25)
        assert w["a"] == 0.8
        w2 = sched.weights_at(0.75)
        assert w2["b"] == 0.7

    def test_curriculum_schedule_weights_at_boundary(self):
        phases = [
            CurriculumPhase("p1", 0.0, 0.5, {"a": 1.0}),
            CurriculumPhase("p2", 0.5, 1.0, {"b": 1.0}),
        ]
        sched = CurriculumSchedule(name="test", phases=phases)
        # At exactly 0.5, should match phase 2 (start_fraction <= t < end_fraction)
        w = sched.weights_at(0.5)
        assert "b" in w

    def test_dataset_profile_defaults(self):
        p = DatasetProfile(name="test")
        assert p.n_tokens == 0
        assert p.n_examples == 0
