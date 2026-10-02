"""``read_raw`` MCP tool implementation."""

from __future__ import annotations

import json

from serum_mcp.preset.raw import RawPreset, get_path

from ._jsonable import jsonable


def read_raw(preset_path: str, path: str | None = None) -> str:
    """Return the full decoded state of a preset as JSON, or the subtree at
    the dotted ``path``."""
    preset = RawPreset.from_file(preset_path)
    if path is None:
        out = {"metadata": preset.metadata, "data": jsonable(preset.data)}
    else:
        out = {"path": path, "value": jsonable(get_path(preset.data, path))}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))
