"""``patch_raw`` MCP tool implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from serum_mcp.preset.raw import RawPreset

from ._jsonable import jsonable


def patch_raw(preset_path: str, patches: dict[str, Any], output_path: str) -> str:
    """Write each ``dotted.path -> value`` in ``patches`` into a copy of the
    preset at ``preset_path`` and save it to ``output_path``. Every other byte
    of the preset is preserved. All patches are checked before anything is
    written.

    Returns the absolute output path as the first line, then one line per
    change (``path: old -> new``), or ``No changes.``.
    """
    patched, changes = RawPreset.from_file(preset_path).patched(patches)
    written = patched.write(output_path)

    lines = [str(Path(written).resolve())]
    for change in changes:
        old = "(absent)" if change.created else jsonable(change.old)
        lines.append(f"{change.path}: {old} -> {jsonable(change.new)}")
    if not changes:
        lines.append("No changes.")
    return "\n".join(lines)
