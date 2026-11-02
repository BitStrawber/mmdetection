from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from .common import existing_file, read_json


REQUIRED_MODEL_FIELDS = ("id", "backbone", "source", "config", "checkpoint")


def load_config(path: str | Path) -> Dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise TypeError("Analysis config must be a JSON object")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Analysis config must contain a non-empty models list")
    seen = set()
    for model in models:
        if not isinstance(model, dict):
            raise TypeError("Each model entry must be a JSON object")
        missing = [field for field in REQUIRED_MODEL_FIELDS if not model.get(field)]
        if missing:
            raise ValueError(
                f"Model {model.get('id', '<unknown>')} is missing {missing}"
            )
        model_id = str(model["id"])
        if model_id in seen:
            raise ValueError(f"Duplicate model id: {model_id}")
        seen.add(model_id)
    return config


def resolve_model_paths(config: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    models = []
    for model in config["models"]:
        item = dict(model)
        item["config"] = str(existing_file(item["config"]))
        item["checkpoint"] = str(existing_file(item["checkpoint"]))
        models.append(item)
    resolved["models"] = models
    return resolved
