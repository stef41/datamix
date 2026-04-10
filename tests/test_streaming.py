"""Tests for datamix.streaming — streaming dataset support."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from datamix.streaming import (
    StreamingDataset,
    stream_interleave,
    stream_jsonl,
)


# ── helpers ───────────────────────────────────────────────────────────────── #


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


SAMPLE_RECORDS = [{"id": i, "text": f"example {i}"} for i in range(10)]


# ── StreamingDataset from iterable ────────────────────────────────────────── #


class TestStreamingDatasetIterable:
    def test_basic_iteration(self):
        ds = StreamingDataset([1, 2, 3])
        assert list(ds) == [1, 2, 3]

    def test_empty_iterable(self):
        ds = StreamingDataset(iter([]))
        assert list(ds) == []

    def test_count(self):
        ds = StreamingDataset(range(5))
        assert ds.count() == 5

    def test_take(self):
        ds = StreamingDataset(range(10))
        assert list(ds.take(3)) == [0, 1, 2]

    def test_take_more_than_available(self):
        ds = StreamingDataset([1, 2])
        assert list(ds.take(10)) == [1, 2]

    def test_skip(self):
        ds = StreamingDataset(range(5))
        assert list(ds.skip(3)) == [3, 4]

    def test_skip_all(self):
        ds = StreamingDataset(range(3))
        assert list(ds.skip(10)) == []

    def test_filter(self):
        ds = StreamingDataset(range(10))
        even = ds.filter(lambda x: x % 2 == 0)
        assert list(even) == [0, 2, 4, 6, 8]

    def test_map(self):
        ds = StreamingDataset([1, 2, 3])
        doubled = ds.map(lambda x: x * 2)
        assert list(doubled) == [2, 4, 6]

    def test_chained_transforms(self):
        ds = StreamingDataset(range(20))
        result = ds.skip(5).take(5).filter(lambda x: x % 2 == 0)
        assert list(result) == [6, 8]

    def test_map_then_filter(self):
        ds = StreamingDataset(range(5))
        result = ds.map(lambda x: x * 10).filter(lambda x: x > 20)
        assert list(result) == [30, 40]


# ── StreamingDataset from JSONL file ──────────────────────────────────────── #


class TestStreamingDatasetFile:
    def test_read_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, SAMPLE_RECORDS)
        ds = StreamingDataset(path)
        items = list(ds)
        assert len(items) == 10
        assert items[0]["id"] == 0

    def test_replayable(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, SAMPLE_RECORDS[:3])
        ds = StreamingDataset(path)
        assert list(ds) == list(ds)  # can iterate twice

    def test_take_from_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, SAMPLE_RECORDS)
        ds = StreamingDataset(path)
        assert len(list(ds.take(3))) == 3

    def test_skip_from_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, SAMPLE_RECORDS)
        ds = StreamingDataset(path)
        items = list(ds.skip(8))
        assert len(items) == 2
        assert items[0]["id"] == 8

    def test_filter_from_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, SAMPLE_RECORDS)
        ds = StreamingDataset(path)
        evens = list(ds.filter(lambda r: r["id"] % 2 == 0))
        assert len(evens) == 5

    def test_count_from_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, SAMPLE_RECORDS)
        ds = StreamingDataset(path)
        assert ds.count() == 10

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            f.write('{"a":1}\n\n{"a":2}\n\n')
        ds = StreamingDataset(path)
        assert ds.count() == 2


# ── stream_jsonl convenience ──────────────────────────────────────────────── #


class TestStreamJsonl:
    def test_stream_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, SAMPLE_RECORDS[:5])
        ds = stream_jsonl(path)
        assert ds.count() == 5


# ── stream_interleave ─────────────────────────────────────────────────────── #


class TestStreamInterleave:
    def test_uniform_interleave(self):
        ds1 = StreamingDataset(["a", "b", "c"])
        ds2 = StreamingDataset([1, 2, 3])
        merged = stream_interleave([ds1, ds2], seed=42)
        items = list(merged)
        assert len(items) == 6
        assert set(items) == {"a", "b", "c", 1, 2, 3}

    def test_weighted_interleave(self):
        ds1 = StreamingDataset(range(100))
        ds2 = StreamingDataset(range(100, 200))
        merged = stream_interleave([ds1, ds2], weights=[0.9, 0.1], seed=0)
        items = list(merged)
        assert len(items) == 200
        # ds1 items should roughly dominate the first portion
        first_50 = items[:50]
        ds1_count = sum(1 for x in first_50 if x < 100)
        assert ds1_count > 25  # should be significantly more than half

    def test_empty_datasets(self):
        merged = stream_interleave([])
        assert list(merged) == []

    def test_single_dataset(self):
        ds = StreamingDataset([1, 2, 3])
        merged = stream_interleave([ds])
        assert list(merged) == [1, 2, 3]

    def test_mismatched_weights_raises(self):
        ds = StreamingDataset([1])
        with pytest.raises(ValueError, match="weights length"):
            stream_interleave([ds], weights=[0.5, 0.5])

    def test_zero_weights_raises(self):
        ds = StreamingDataset([1])
        with pytest.raises(ValueError, match="Sum of weights"):
            stream_interleave([ds], weights=[0.0])
