"""Tests for datamix.quality."""

import pytest
from datamix.quality import dedup_exact, dedup_ngram, length_filter, quality_score


class TestLengthFilter:
    def test_min_length(self):
        examples = ["hi", "hello world this is a longer text", "ok"]
        kept, stats = length_filter(examples, min_length=10)
        assert stats["kept"] == 1
        assert stats["removed"] == 2

    def test_max_length(self):
        examples = ["short", "a" * 1000]
        kept, stats = length_filter(examples, max_length=100)
        assert stats["kept"] == 1

    def test_min_words(self):
        examples = ["one", "one two three four five six"]
        kept, stats = length_filter(examples, min_words=3)
        assert stats["kept"] == 1

    def test_max_words(self):
        examples = ["one two", "one two three four five six seven eight nine ten"]
        kept, stats = length_filter(examples, max_words=5)
        assert stats["kept"] == 1

    def test_no_filter(self):
        examples = ["a", "b", "c"]
        kept, stats = length_filter(examples)
        assert stats["kept"] == 3

    def test_empty(self):
        kept, stats = length_filter([])
        assert stats["total"] == 0


class TestDedupExact:
    def test_removes_dupes(self):
        examples = ["hello", "world", "hello", "hello"]
        kept, stats = dedup_exact(examples)
        assert stats["kept"] == 2
        assert stats["duplicates_removed"] == 2

    def test_no_dupes(self):
        examples = ["a", "b", "c"]
        kept, stats = dedup_exact(examples)
        assert stats["kept"] == 3
        assert stats["duplicates_removed"] == 0

    def test_preserves_order(self):
        examples = ["first", "second", "first"]
        kept, _ = dedup_exact(examples)
        assert kept == ["first", "second"]


class TestDedupNgram:
    def test_near_duplicates(self):
        examples = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy cat",  # 1 word different
            "completely different text about machine learning",
        ]
        kept, stats = dedup_ngram(examples, n=3, threshold=0.7)
        # The first two should be flagged as near-dupes
        assert stats["kept"] <= 2

    def test_no_near_dupes(self):
        examples = [
            "the quick brown fox",
            "machine learning is great",
            "python programming language",
        ]
        kept, stats = dedup_ngram(examples, n=3, threshold=0.8)
        assert stats["kept"] == 3

    def test_empty(self):
        kept, stats = dedup_ngram([])
        assert stats["kept"] == 0


class TestQualityScore:
    def test_good_text(self):
        text = "Machine learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data."
        score = quality_score(text)
        assert score > 0.5

    def test_empty(self):
        assert quality_score("") == 0.0

    def test_short_text(self):
        score = quality_score("hi")
        assert score < 0.5

    def test_repetitive(self):
        text = "the the the the the the the the the the the the"
        score = quality_score(text, min_length=10)
        assert score < 0.8

    def test_non_text(self):
        text = "!!!@@@###$$$%%%^^^&&&***((())))" * 3
        score = quality_score(text, min_length=10)
        assert score < 0.5

    def test_range(self):
        texts = ["hi", "Hello, this is a well-structured sentence with good grammar.", "", "x" * 1000]
        for text in texts:
            score = quality_score(text)
            assert 0.0 <= score <= 1.0
