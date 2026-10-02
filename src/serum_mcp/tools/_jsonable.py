"""JSON rendering of raw decoded preset data, shared by read_raw and patch_raw."""

from __future__ import annotations

from typing import Any

import numpy as np

from serum_mcp.preset.raw import Float64


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
