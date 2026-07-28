"""Provenance stamping.

Every artefact this project produces carries a header recording exactly what
produced it: the git commit, whether the tree was dirty, the config hash, seeds,
model identifiers, and the wall-clock time.

This is not bureaucracy. A measurement study whose numbers cannot be traced to a
specific commit and configuration is not reproducible, and a reviewer is entitled
to disbelieve it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_sha() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


@dataclass
class Provenance:
    """A record of what produced an artefact."""

    script: str
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    git_sha: str | None = field(default_factory=git_sha)
    git_dirty: bool | None = field(default_factory=git_dirty)
    config_hash: str | None = None
    config_path: str | None = None
    seed: int | None = None
    models: dict[str, str] = field(default_factory=dict)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stamp(
    script: str,
    config: Any = None,
    config_path: str | None = None,
    seed: int | None = None,
    models: dict[str, str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a provenance dict for a JSONL header.

    Warns loudly on a dirty tree: an artefact produced from uncommitted code
    cannot be regenerated.
    """
    prov = Provenance(
        script=script,
        config_hash=getattr(config, "content_hash", None) if config is not None else None,
        config_path=config_path,
        seed=seed,
        models=models or {},
        extra=extra,
    )
    if prov.git_dirty:
        print(
            "WARNING: git tree is dirty. This artefact was produced from uncommitted "
            "code and cannot be reproduced from the recorded commit. Commit first.",
            file=sys.stderr,
        )
    return prov.to_dict()
