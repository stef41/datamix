"""Tests for datamix.curriculum."""

import pytest
from datamix.curriculum import cosine_schedule, custom_schedule, linear_schedule, step_schedule
from datamix._types import CurriculumPhase


DATASETS = ["wiki", "code", "books"]


class TestLinearSchedule:
    def test_basic(self):
        sched = linear_schedule(DATASETS, n_phases=3)
        assert sched.n_phases == 3
        assert sched.name == "linear"

    def test_phases_cover_range(self):
        sched = linear_schedule(DATASETS, n_phases=4)
        assert sched.phases[0].start_fraction == pytest.approx(0.0)
        assert sched.phases[-1].end_fraction == pytest.approx(1.0)

    def test_weights_have_all_datasets(self):
        sched = linear_schedule(DATASETS, n_phases=2)
        for phase in sched.phases:
            assert set(phase.weights.keys()) == set(DATASETS)

    def test_custom_start_end(self):
        sched = linear_schedule(
            DATASETS, n_phases=2,
            start_weights={"wiki": 0.8, "code": 0.1, "books": 0.1},
            end_weights={"wiki": 0.2, "code": 0.4, "books": 0.4},
        )
        # First phase should be closer to start weights
        assert sched.phases[0].weights["wiki"] > sched.phases[-1].weights["wiki"]

    def test_empty(self):
        sched = linear_schedule([])
        assert sched.n_phases == 0


class TestCosineSchedule:
    def test_basic(self):
        sched = cosine_schedule(DATASETS, n_phases=4)
        assert sched.n_phases == 4
        assert sched.name == "cosine"

    def test_primary_decays(self):
        sched = cosine_schedule(DATASETS, n_phases=4, primary="wiki")
        # Primary weight should decrease over phases
        first_w = sched.phases[0].weights["wiki"]
        last_w = sched.phases[-1].weights["wiki"]
        assert first_w > last_w

    def test_default_primary(self):
        sched = cosine_schedule(DATASETS, n_phases=2)
        # Default primary is first dataset
        assert "wiki" in sched.phases[0].weights

    def test_empty(self):
        sched = cosine_schedule([])
        assert sched.n_phases == 0


class TestStepSchedule:
    def test_basic(self):
        config = [
            {"name": "warmup", "fraction": 0.1, "weights": {"wiki": 0.8, "code": 0.2}},
            {"name": "main", "fraction": 0.7, "weights": {"wiki": 0.5, "code": 0.5}},
            {"name": "cooldown", "fraction": 0.2, "weights": {"wiki": 0.3, "code": 0.7}},
        ]
        sched = step_schedule(config)
        assert sched.n_phases == 3

    def test_phases_contiguous(self):
        config = [
            {"name": "p1", "fraction": 0.5, "weights": {}},
            {"name": "p2", "fraction": 0.5, "weights": {}},
        ]
        sched = step_schedule(config)
        assert sched.phases[0].end_fraction == pytest.approx(sched.phases[1].start_fraction)


class TestCustomSchedule:
    def test_basic(self):
        phases = [CurriculumPhase("p1", 0.0, 0.5, {"a": 1.0})]
        sched = custom_schedule(phases, name="my_sched")
        assert sched.name == "my_sched"
        assert sched.n_phases == 1
