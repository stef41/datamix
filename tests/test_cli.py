"""Tests for datamix.cli."""

import json
import pytest
from click.testing import CliRunner
from datamix.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def jsonl_file(tmp_path):
    path = tmp_path / "data.jsonl"
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a powerful tool.",
        "Python is great for data science.",
        "Deep learning models need lots of data.",
        "Natural language processing is fascinating.",
    ]
    with open(path, "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
    return str(path)


class TestProfileCommand:
    def test_basic(self, runner, jsonl_file):
        result = runner.invoke(cli, ["profile", jsonl_file])
        assert result.exit_code == 0

    def test_max_examples(self, runner, jsonl_file):
        result = runner.invoke(cli, ["profile", jsonl_file, "--max-examples", "2"])
        assert result.exit_code == 0


class TestMixCommand:
    def test_basic(self, runner, jsonl_file):
        result = runner.invoke(cli, ["mix", jsonl_file, jsonl_file])
        assert result.exit_code == 0

    def test_equal_strategy(self, runner, jsonl_file):
        result = runner.invoke(cli, ["mix", jsonl_file, "--strategy", "equal"])
        assert result.exit_code == 0


class TestCleanCommand:
    def test_basic(self, runner, jsonl_file):
        result = runner.invoke(cli, ["clean", jsonl_file])
        assert result.exit_code == 0
        assert "Loaded" in result.output

    def test_min_length(self, runner, jsonl_file):
        result = runner.invoke(cli, ["clean", jsonl_file, "--min-length", "100"])
        assert result.exit_code == 0
