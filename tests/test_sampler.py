"""Tests for datamix.sampler."""

import pytest
from datamix.sampler import proportional_sample, stratified_sample, temperature_sample


DATASETS = {
    "wiki": ["wiki " + str(i) for i in range(100)],
    "code": ["code " + str(i) for i in range(50)],
    "books": ["books " + str(i) for i in range(20)],
}


class TestTemperatureSample:
    def test_basic(self):
        result = temperature_sample(DATASETS, n_samples=50)
        assert len(result) == 50

    def test_deterministic(self):
        r1 = temperature_sample(DATASETS, n_samples=20, seed=42)
        r2 = temperature_sample(DATASETS, n_samples=20, seed=42)
        assert r1 == r2

    def test_different_seeds(self):
        r1 = temperature_sample(DATASETS, n_samples=20, seed=1)
        r2 = temperature_sample(DATASETS, n_samples=20, seed=2)
        assert r1 != r2

    def test_high_temperature_uniform(self):
        # Very high temp should sample more uniformly
        result = temperature_sample(DATASETS, n_samples=1000, temperature=100.0, seed=42)
        wiki_count = sum(1 for r in result if r.startswith("wiki"))
        code_count = sum(1 for r in result if r.startswith("code"))
        books_count = sum(1 for r in result if r.startswith("books"))
        # With very high temp, counts should be roughly equal (within 50% of each other)
        counts = [wiki_count, code_count, books_count]
        assert max(counts) < 2 * min(counts)

    def test_low_temperature(self):
        result = temperature_sample(DATASETS, n_samples=1000, temperature=0.1, seed=42)
        wiki_count = sum(1 for r in result if r.startswith("wiki"))
        # Low temp amplifies largest dataset
        assert wiki_count > 500

    def test_empty(self):
        result = temperature_sample({}, n_samples=10)
        assert result == []


class TestProportionalSample:
    def test_basic(self):
        result = proportional_sample(DATASETS, n_samples=50)
        assert len(result) == 50

    def test_larger_dataset_more_samples(self):
        result = proportional_sample(DATASETS, n_samples=1000, seed=42)
        wiki_count = sum(1 for r in result if r.startswith("wiki"))
        books_count = sum(1 for r in result if r.startswith("books"))
        assert wiki_count > books_count


class TestStratifiedSample:
    def test_basic(self):
        examples = ["short " + str(i) for i in range(50)] + ["longer text " + str(i) for i in range(50)]
        cat_fn = lambda x: "short" if x.startswith("short") else "long"
        result = stratified_sample(examples, cat_fn, n_samples=20)
        assert len(result) == 20

    def test_balanced(self):
        examples = ["cat " + str(i) for i in range(100)] + ["dog " + str(i) for i in range(10)]
        cat_fn = lambda x: x.split()[0]
        result = stratified_sample(examples, cat_fn, n_samples=20)
        cat_count = sum(1 for r in result if r.startswith("cat"))
        dog_count = sum(1 for r in result if r.startswith("dog"))
        # Should be more balanced than proportional
        assert dog_count >= 5
