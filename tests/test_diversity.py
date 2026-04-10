"""Tests for datamix.diversity."""

import math
import pytest

from datamix.diversity import (
    DiversityAnalyzer,
    DiversityMetrics,
    format_diversity_report,
)


@pytest.fixture
def analyzer():
    return DiversityAnalyzer(ngram_size=3)


@pytest.fixture
def diverse_texts():
    return [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming every industry",
        "Quantum computing may revolutionize cryptography",
        "The ocean contains vast unexplored ecosystems",
        "Music theory explores harmony rhythm and melody",
    ]


@pytest.fixture
def repetitive_texts():
    return [
        "The cat sat on the mat",
        "The cat sat on the mat",
        "The cat sat on the mat again",
    ]


# ── DiversityMetrics ─────────────────────────────────────────────────────────


class TestDiversityMetrics:
    def test_dataclass_fields(self):
        m = DiversityMetrics(
            vocabulary_richness=0.8,
            length_variance=10.0,
            topic_entropy=3.5,
            unique_ngram_ratio=0.9,
            redundancy_score=0.1,
        )
        assert m.vocabulary_richness == 0.8
        assert m.redundancy_score == 0.1


# ── DiversityAnalyzer.analyze ────────────────────────────────────────────────


class TestAnalyze:
    def test_returns_metrics(self, analyzer, diverse_texts):
        m = analyzer.analyze(diverse_texts)
        assert isinstance(m, DiversityMetrics)

    def test_empty_input(self, analyzer):
        m = analyzer.analyze([])
        assert m.vocabulary_richness == 0.0
        assert m.length_variance == 0.0
        assert m.topic_entropy == 0.0
        assert m.unique_ngram_ratio == 0.0
        assert m.redundancy_score == 0.0


# ── vocabulary_richness ──────────────────────────────────────────────────────


class TestVocabularyRichness:
    def test_all_unique(self, analyzer):
        texts = ["one two three four five six seven eight"]
        vr = analyzer.vocabulary_richness(texts)
        assert vr == 1.0

    def test_all_same(self, analyzer):
        texts = ["the the the the the"]
        vr = analyzer.vocabulary_richness(texts)
        assert abs(vr - 0.2) < 1e-6

    def test_empty(self, analyzer):
        assert analyzer.vocabulary_richness([]) == 0.0


# ── length_distribution ──────────────────────────────────────────────────────


class TestLengthDistribution:
    def test_basic_stats(self, analyzer):
        texts = ["abc", "abcdef", "abcdefghi"]  # lengths 3, 6, 9
        dist = analyzer.length_distribution(texts)
        assert dist["mean"] == 6.0
        assert dist["min"] == 3.0
        assert dist["max"] == 9.0
        assert dist["median"] == 6.0
        assert dist["std"] > 0

    def test_single_text(self, analyzer):
        dist = analyzer.length_distribution(["hello"])
        assert dist["mean"] == 5.0
        assert dist["std"] == 0.0

    def test_empty(self, analyzer):
        dist = analyzer.length_distribution([])
        assert dist["mean"] == 0.0


# ── ngram_diversity ──────────────────────────────────────────────────────────


class TestNgramDiversity:
    def test_unique_ngrams_high(self, analyzer, diverse_texts):
        ratio = analyzer.ngram_diversity(diverse_texts)
        assert ratio > 0.8

    def test_repetitive_ngrams_lower(self, analyzer, repetitive_texts):
        ratio = analyzer.ngram_diversity(repetitive_texts)
        assert ratio < 0.8

    def test_custom_n(self, analyzer, diverse_texts):
        ratio = analyzer.ngram_diversity(diverse_texts, n=2)
        assert 0.0 < ratio <= 1.0

    def test_empty(self, analyzer):
        assert analyzer.ngram_diversity([]) == 0.0


# ── topic_entropy ────────────────────────────────────────────────────────────


class TestTopicEntropy:
    def test_diverse_higher(self, analyzer, diverse_texts, repetitive_texts):
        e_div = analyzer.topic_entropy(diverse_texts)
        e_rep = analyzer.topic_entropy(repetitive_texts)
        assert e_div > e_rep

    def test_empty(self, analyzer):
        assert analyzer.topic_entropy([]) == 0.0


# ── redundancy_score ─────────────────────────────────────────────────────────


class TestRedundancyScore:
    def test_repetitive_high(self, analyzer, repetitive_texts):
        score = analyzer.redundancy_score(repetitive_texts)
        assert score > 0.2

    def test_diverse_low(self, analyzer, diverse_texts):
        score = analyzer.redundancy_score(diverse_texts)
        assert score < 0.5

    def test_single_text(self, analyzer):
        assert analyzer.redundancy_score(["only one"]) == 0.0


# ── compare_diversity ────────────────────────────────────────────────────────


class TestCompareDiversity:
    def test_keys(self, analyzer, diverse_texts, repetitive_texts):
        result = analyzer.compare_diversity(diverse_texts, repetitive_texts)
        assert set(result.keys()) == {"a", "b", "delta"}
        assert "vocabulary_richness" in result["a"]
        assert "vocabulary_richness" in result["delta"]

    def test_delta_sign(self, analyzer, diverse_texts, repetitive_texts):
        result = analyzer.compare_diversity(diverse_texts, repetitive_texts)
        # Diverse texts should have higher vocabulary richness
        assert result["delta"]["vocabulary_richness"] > 0


# ── format_diversity_report ──────────────────────────────────────────────────


class TestFormatReport:
    def test_contains_all_fields(self):
        m = DiversityMetrics(
            vocabulary_richness=0.75,
            length_variance=120.5,
            topic_entropy=4.2,
            unique_ngram_ratio=0.88,
            redundancy_score=0.12,
        )
        report = format_diversity_report(m)
        assert "Vocabulary richness" in report
        assert "0.7500" in report
        assert "Topic entropy" in report
        assert "Redundancy score" in report

    def test_report_is_string(self, analyzer, diverse_texts):
        m = analyzer.analyze(diverse_texts)
        report = format_diversity_report(m)
        assert isinstance(report, str)
        assert len(report) > 50
