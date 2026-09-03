"""Deterministic signatures for simulation-defining inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(parameters: Mapping[str, Any]) -> str:
    """Return the stable JSON representation used to create signatures."""

    return json.dumps(
        _json_value(parameters),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def create_run_signature(parameters: Mapping[str, Any]) -> str:
    """Return a lowercase, 64-character SHA-256 signature."""

    payload = canonical_json(parameters).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

