"""Tests for datamix.mixer."""

import pytest
from datamix._types import DatasetProfile, MixRecipe, MixStrategy
from datamix.mixer import create_recipe, merge_recipes, scale_recipe


def _profiles():
    return [
        DatasetProfile(name="wiki", n_examples=1000, n_tokens=50000),
        DatasetProfile(name="code", n_examples=500, n_tokens=30000),
        DatasetProfile(name="books", n_examples=200, n_tokens=20000),
    ]


class TestCreateRecipe:
    def test_proportional(self):
        r = create_recipe(_profiles())
        assert r.strategy == MixStrategy.PROPORTIONAL
        assert r.n_components == 3
        # wiki has most tokens, should have highest weight
        nw = r.normalized_weights
        assert nw["wiki"] > nw["code"]

    def test_equal(self):
        r = create_recipe(_profiles(), strategy=MixStrategy.EQUAL)
        nw = r.normalized_weights
        assert abs(nw["wiki"] - nw["code"]) < 0.01

    def test_temperature(self):
        # High temperature → more uniform
        r_low = create_recipe(_profiles(), strategy=MixStrategy.TEMPERATURE, temperature=0.5)
        r_high = create_recipe(_profiles(), strategy=MixStrategy.TEMPERATURE, temperature=5.0)
        # High temp should be more uniform (smaller spread)
        nw_low = r_low.normalized_weights
        nw_high = r_high.normalized_weights
        spread_low = max(nw_low.values()) - min(nw_low.values())
        spread_high = max(nw_high.values()) - min(nw_high.values())
        assert spread_high < spread_low

    def test_custom(self):
        weights = {"wiki": 0.5, "code": 0.3, "books": 0.2}
        r = create_recipe(_profiles(), strategy=MixStrategy.CUSTOM, weights=weights)
        assert r.components == weights

    def test_empty(self):
        r = create_recipe([])
        assert r.n_components == 0

    def test_total_tokens(self):
        r = create_recipe(_profiles(), total_tokens=1000000)
        assert r.total_tokens == 1000000


class TestMergeRecipes:
    def test_equal_merge(self):
        r1 = MixRecipe(components={"a": 0.8, "b": 0.2})
        r2 = MixRecipe(components={"a": 0.4, "c": 0.6})
        merged = merge_recipes([r1, r2])
        assert "a" in merged.components
        assert "b" in merged.components
        assert "c" in merged.components

    def test_weighted_merge(self):
        r1 = MixRecipe(components={"a": 1.0})
        r2 = MixRecipe(components={"b": 1.0})
        merged = merge_recipes([r1, r2], weights=[0.8, 0.2])
        assert merged.components["a"] > merged.components["b"]

    def test_empty(self):
        r = merge_recipes([])
        assert r.n_components == 0

    def test_mismatched_weights(self):
        r1 = MixRecipe(components={"a": 1.0})
        with pytest.raises(ValueError):
            merge_recipes([r1], weights=[0.5, 0.5])


class TestScaleRecipe:
    def test_double(self):
        r = MixRecipe(components={"a": 0.5}, total_tokens=1000)
        scaled = scale_recipe(r, 2.0)
        assert scaled.total_tokens == 2000
        assert scaled.components == r.components

    def test_half(self):
        r = MixRecipe(components={"a": 0.5}, total_tokens=1000)
        scaled = scale_recipe(r, 0.5)
        assert scaled.total_tokens == 500
