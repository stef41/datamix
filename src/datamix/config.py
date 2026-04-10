"""Serialize / deserialize MixRecipe to YAML and JSON config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from datamix._types import DatamixError, MixRecipe

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _recipe_to_dict(recipe: MixRecipe) -> Dict[str, Any]:
    """Convert a MixRecipe to the portable config dict format."""
    sources = [
        {"path": path, "weight": weight}
        for path, weight in recipe.components.items()
    ]
    d: Dict[str, Any] = {"name": recipe.name, "sources": sources}
    if recipe.total_tokens:
        d["token_budget"] = recipe.total_tokens
    if recipe.metadata:
        d["metadata"] = recipe.metadata
    return d


def _dict_to_recipe(d: Dict[str, Any]) -> MixRecipe:
    """Build a MixRecipe from a portable config dict."""
    if not isinstance(d, dict):
        raise DatamixError("config must be a mapping")
    sources = d.get("sources", [])
    components: Dict[str, float] = {}
    for src in sources:
        path = src.get("path")
        weight = src.get("weight")
        if path is None or weight is None:
            raise DatamixError(f"each source must have 'path' and 'weight', got {src!r}")
        components[str(path)] = float(weight)
    return MixRecipe(
        name=d.get("name", "mix"),
        components=components,
        total_tokens=int(d.get("token_budget", 0)),
        metadata=d.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def save_recipe_yaml(recipe: MixRecipe, path: Union[str, Path]) -> Path:
    """Save a MixRecipe to a YAML file.

    Requires PyYAML (``pip install pyyaml``).  Raises *DatamixError* if
    PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise DatamixError(
            "PyYAML is required for YAML support: pip install pyyaml"
        ) from exc

    path = Path(path)
    data = _recipe_to_dict(recipe)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def load_recipe_yaml(path: Union[str, Path]) -> MixRecipe:
    """Load a MixRecipe from a YAML file.

    Requires PyYAML (``pip install pyyaml``).  Raises *DatamixError* if
    PyYAML is not installed or the file is invalid.
    """
    try:
        import yaml
    except ImportError as exc:
        raise DatamixError(
            "PyYAML is required for YAML support: pip install pyyaml"
        ) from exc

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        raise DatamixError(f"empty YAML file: {path}")
    return _dict_to_recipe(data)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_recipe_json(recipe: MixRecipe, path: Union[str, Path]) -> Path:
    """Save a MixRecipe to a JSON file."""
    path = Path(path)
    data = _recipe_to_dict(recipe)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def load_recipe_json(path: Union[str, Path]) -> MixRecipe:
    """Load a MixRecipe from a JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return _dict_to_recipe(data)
