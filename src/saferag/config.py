"""Configuration loading.

Configs are plain YAML. They are hashed on load so that every output file can
record exactly which configuration produced it -- see utils.provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict with attribute access and a stable content hash.

    >>> c = Config({"a": {"b": 1}})
    >>> c.a.b
    1
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @property
    def content_hash(self) -> str:
        """Deterministic hash of the config contents."""
        blob = yaml.safe_dump(_plain(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def load_config(path: str | Path) -> Config:
    """Load a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}\n"
            "Run from the repository root, or pass --config with an explicit path."
        )
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return Config(data)


def project_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]
