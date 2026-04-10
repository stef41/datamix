"""Tests for datamix.config — YAML and JSON recipe serialization."""

from __future__ import annotations

import json

import pytest

from datamix._types import DatamixError, MixRecipe
from datamix.config import (
    load_recipe_json,
    load_recipe_yaml,
    save_recipe_json,
    save_recipe_yaml,
    _dict_to_recipe,
    _recipe_to_dict,
)

yaml = pytest.importorskip("yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_recipe() -> MixRecipe:
    return MixRecipe(
        name="my-mix",
        components={
            "data/code.jsonl": 0.4,
            "data/math.jsonl": 0.3,
            "data/general.jsonl": 0.3,
        },
        total_tokens=1_000_000,
    )


# ---------------------------------------------------------------------------
# Round-trip: YAML
# ---------------------------------------------------------------------------

def test_yaml_roundtrip(tmp_path, sample_recipe):
    p = tmp_path / "recipe.yaml"
    save_recipe_yaml(sample_recipe, p)
    loaded = load_recipe_yaml(p)

    assert loaded.name == sample_recipe.name
    assert loaded.components == sample_recipe.components
    assert loaded.total_tokens == sample_recipe.total_tokens


def test_yaml_file_content(tmp_path, sample_recipe):
    """The YAML file should be human-readable with the expected keys."""
    p = tmp_path / "recipe.yaml"
    save_recipe_yaml(sample_recipe, p)
    data = yaml.safe_load(p.read_text())

    assert data["name"] == "my-mix"
    assert len(data["sources"]) == 3
    assert data["token_budget"] == 1_000_000
    for src in data["sources"]:
        assert "path" in src
        assert "weight" in src


# ---------------------------------------------------------------------------
# Round-trip: JSON
# ---------------------------------------------------------------------------

def test_json_roundtrip(tmp_path, sample_recipe):
    p = tmp_path / "recipe.json"
    save_recipe_json(sample_recipe, p)
    loaded = load_recipe_json(p)

    assert loaded.name == sample_recipe.name
    assert loaded.components == sample_recipe.components
    assert loaded.total_tokens == sample_recipe.total_tokens


def test_json_file_content(tmp_path, sample_recipe):
    p = tmp_path / "recipe.json"
    save_recipe_json(sample_recipe, p)
    data = json.loads(p.read_text())

    assert data["name"] == "my-mix"
    assert len(data["sources"]) == 3
    assert data["token_budget"] == 1_000_000


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_recipe_roundtrip_json(tmp_path):
    recipe = MixRecipe()
    p = tmp_path / "empty.json"
    save_recipe_json(recipe, p)
    loaded = load_recipe_json(p)
    assert loaded.name == "mix"
    assert loaded.components == {}
    assert loaded.total_tokens == 0


def test_empty_recipe_roundtrip_yaml(tmp_path):
    recipe = MixRecipe()
    p = tmp_path / "empty.yaml"
    save_recipe_yaml(recipe, p)
    loaded = load_recipe_yaml(p)
    assert loaded.name == "mix"
    assert loaded.components == {}


def test_missing_weight_raises():
    with pytest.raises(DatamixError, match="weight"):
        _dict_to_recipe({"sources": [{"path": "a.jsonl"}]})


def test_missing_path_raises():
    with pytest.raises(DatamixError, match="path"):
        _dict_to_recipe({"sources": [{"weight": 0.5}]})


def test_non_dict_raises():
    with pytest.raises(DatamixError, match="mapping"):
        _dict_to_recipe("not a dict")  # type: ignore[arg-type]


def test_load_yaml_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(DatamixError, match="empty YAML"):
        load_recipe_yaml(p)


def test_load_json_invalid(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        load_recipe_json(p)


def test_missing_optional_fields():
    """token_budget and metadata are optional — defaults should apply."""
    recipe = _dict_to_recipe({"name": "bare", "sources": []})
    assert recipe.total_tokens == 0
    assert recipe.metadata == {}
    assert recipe.components == {}


def test_metadata_preserved(tmp_path):
    recipe = MixRecipe(
        name="meta",
        components={"a.jsonl": 1.0},
        total_tokens=500,
        metadata={"author": "test"},
    )
    p = tmp_path / "meta.json"
    save_recipe_json(recipe, p)
    loaded = load_recipe_json(p)
    assert loaded.metadata == {"author": "test"}


def test_recipe_to_dict_no_token_budget():
    """When total_tokens is 0, token_budget key should be absent."""
    recipe = MixRecipe(name="x", components={"a": 0.5})
    d = _recipe_to_dict(recipe)
    assert "token_budget" not in d
