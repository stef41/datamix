"""Tests for datamix.profile."""

import json
import pytest
from pathlib import Path
from datamix._types import DatasetProfile
from datamix.profile import compare_profiles, profile_dataset, profile_jsonl


SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Neural networks can learn complex patterns from data.",
    "Natural language processing enables computers to understand text.",
]


class TestProfileDataset:
    def test_basic(self):
        p = profile_dataset(SAMPLE_TEXTS, name="test")
        assert p.name == "test"
        assert p.n_examples == 5
        assert p.n_tokens > 0
        assert p.avg_tokens_per_example > 0

    def test_quality_metrics(self):
        p = profile_dataset(SAMPLE_TEXTS, name="test")
        assert p.quality.avg_length > 0
        assert p.quality.min_length > 0
        assert p.quality.max_length >= p.quality.min_length

    def test_empty(self):
        p = profile_dataset([], name="empty")
        assert p.n_examples == 0
        assert p.n_tokens == 0

    def test_custom_tokenizer(self):
        tokenizer = lambda text: text.split()
        p = profile_dataset(SAMPLE_TEXTS, name="test", tokenizer=tokenizer)
        assert p.n_tokens > 0

    def test_category_fn(self):
        cat_fn = lambda text: "short" if len(text) < 50 else "long"
        p = profile_dataset(SAMPLE_TEXTS, name="test", category_fn=cat_fn)
        assert len(p.categories) > 0
        assert sum(p.categories.values()) == 5

    def test_single_example(self):
        p = profile_dataset(["hello world"], name="single")
        assert p.n_examples == 1
        assert p.quality.std_length == 0.0


class TestProfileJsonl:
    def test_basic(self, tmp_path):
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            for text in SAMPLE_TEXTS:
                f.write(json.dumps({"text": text}) + "\n")
        p = profile_jsonl(path)
        assert p.n_examples == 5
        assert p.n_tokens > 0

    def test_custom_key(self, tmp_path):
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            for text in SAMPLE_TEXTS:
                f.write(json.dumps({"content": text}) + "\n")
        p = profile_jsonl(path, text_key="content")
        assert p.n_examples == 5

    def test_max_examples(self, tmp_path):
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            for text in SAMPLE_TEXTS:
                f.write(json.dumps({"text": text}) + "\n")
        p = profile_jsonl(path, max_examples=2)
        assert p.n_examples == 2

    def test_not_found(self, tmp_path):
        from datamix._types import DatamixError
        with pytest.raises(DatamixError):
            profile_jsonl(tmp_path / "missing.jsonl")

    def test_name_from_file(self, tmp_path):
        path = tmp_path / "wikipedia.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
        p = profile_jsonl(path)
        assert p.name == "wikipedia"


class TestCompareProfiles:
    def test_basic(self):
        profiles = [
            DatasetProfile(name="a", n_examples=100, n_tokens=5000),
            DatasetProfile(name="b", n_examples=200, n_tokens=10000),
        ]
        result = compare_profiles(profiles)
        assert result["n_datasets"] == 2
        assert result["total_tokens"] == 15000
        assert result["total_examples"] == 300

    def test_empty(self):
        result = compare_profiles([])
        assert result["total_tokens"] == 0
