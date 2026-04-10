"""Tests for datamix._types."""

import pytest
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


class TestDatasetProfile:
    def test_basic(self):
        p = DatasetProfile(name="test", n_examples=1000, n_tokens=50000)
        assert p.size_tokens_m == pytest.approx(0.05)

    def test_to_dict(self):
        p = DatasetProfile(name="test", n_examples=100, n_tokens=5000)
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["n_examples"] == 100

    def test_from_dict(self):
        d = {"name": "test", "n_examples": 100, "n_tokens": 5000}
        p = DatasetProfile.from_dict(d)
        assert p.name == "test"
        assert p.n_examples == 100

    def test_roundtrip(self):
        p = DatasetProfile(name="test", n_examples=100, n_tokens=5000)
        d = p.to_dict()
        p2 = DatasetProfile.from_dict(d)
        assert p2.name == p.name

    def test_empty(self):
        p = DatasetProfile(name="empty")
        assert p.size_tokens_m == 0.0


class TestMixRecipe:
    def test_basic(self):
        r = MixRecipe(components={"a": 0.7, "b": 0.3})
        assert r.n_components == 2

    def test_normalized_weights(self):
        r = MixRecipe(components={"a": 70, "b": 30})
        nw = r.normalized_weights
        assert nw["a"] == pytest.approx(0.7)
        assert nw["b"] == pytest.approx(0.3)

    def test_empty(self):
        r = MixRecipe()
        assert r.n_components == 0
        assert r.normalized_weights == {}

    def test_to_dict(self):
        r = MixRecipe(name="test", components={"a": 0.5, "b": 0.5})
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["n_components"] == 2

    def test_from_dict(self):
        d = {"name": "test", "components": {"a": 0.5}, "strategy": "proportional"}
        r = MixRecipe.from_dict(d)
        assert r.name == "test"
        assert r.strategy == MixStrategy.PROPORTIONAL


class TestCurriculumSchedule:
    def test_weights_at(self):
        phases = [
            CurriculumPhase("warmup", 0.0, 0.3, {"a": 0.8, "b": 0.2}),
            CurriculumPhase("main", 0.3, 1.0, {"a": 0.5, "b": 0.5}),
        ]
        sched = CurriculumSchedule(phases=phases)
        assert sched.weights_at(0.1) == {"a": 0.8, "b": 0.2}
        assert sched.weights_at(0.5) == {"a": 0.5, "b": 0.5}

    def test_n_phases(self):
        phases = [
            CurriculumPhase("p1", 0.0, 0.5, {}),
            CurriculumPhase("p2", 0.5, 1.0, {}),
        ]
        sched = CurriculumSchedule(phases=phases)
        assert sched.n_phases == 2

    def test_to_dict(self):
        sched = CurriculumSchedule(name="test")
        d = sched.to_dict()
        assert d["name"] == "test"

    def test_empty_weights_at(self):
        sched = CurriculumSchedule()
        assert sched.weights_at(0.5) == {}


class TestCurriculumPhase:
    def test_duration(self):
        p = CurriculumPhase("p", 0.2, 0.6, {})
        assert p.duration == pytest.approx(0.4)


class TestTokenBudget:
    def test_to_dict(self):
        b = TokenBudget(total_tokens=1000, allocations={"a": 600, "b": 400})
        d = b.to_dict()
        assert d["total_tokens"] == 1000
        assert d["n_datasets"] == 2


class TestDatamixError:
    def test_is_exception(self):
        assert issubclass(DatamixError, Exception)


class TestQualityMetrics:
    def test_defaults(self):
        q = QualityMetrics()
        assert q.avg_length == 0.0
        assert q.n_duplicates == 0


class TestSamplingConfig:
    def test_defaults(self):
        c = SamplingConfig()
        assert c.seed == 42
        assert c.temperature == 1.0
