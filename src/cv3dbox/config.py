from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if "extends" in config:
        base = load_config(path.parent / config.pop("extends"))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        return base
    return config


def resolve_project_path(project_root: Path, value: str | None) -> Path | None:
    """Resolve a path from the project root unless it is already absolute."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path
