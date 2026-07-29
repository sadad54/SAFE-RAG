"""Console logging with a consistent format across scripts."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "saferag", level: int = logging.INFO) -> logging.Logger:
    """Return a logger under the ``saferag`` namespace.

    The namespacing is load-bearing, not cosmetic. Only the ``saferag`` logger is
    configured, so ``getLogger("run_rag")`` returns an unparented logger whose
    INFO records fall through to the root logger and are dropped at its default
    WARNING level. Every log.info in the pipeline was invisible until this.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("saferag")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    if name != "saferag" and not name.startswith("saferag."):
        name = f"saferag.{name}"
    return logging.getLogger(name)
