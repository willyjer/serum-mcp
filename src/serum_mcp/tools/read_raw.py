"""``read_raw`` MCP tool implementation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from serum_mcp.preset.raw import Float64, RawPreset, get_path


def jsonable(value: Any) -> Any:
    """Make decoded data JSON-safe, showing each 32-bit float as the shortest
    decimal that reads back to the same 32-bit value (0.3, not
    0.30000001192092896)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, Float64):
        return float(value)
    if isinstance(value, float):
        return float(str(np.float32(value)))
    if isinstance(value, bytes):
        return {"$bytes": len(value)}
    return value


def read_raw(preset_path: str, path: str | None = None) -> str:
    """Return the full decoded state of a preset as JSON, or the subtree at
    the dotted ``path``."""
    preset = RawPreset.from_file(preset_path)
    if path is None:
        out = {"metadata": preset.metadata, "data": jsonable(preset.data)}
    else:
        out = {"path": path, "value": jsonable(get_path(preset.data, path))}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))
