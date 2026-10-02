"""Binary (un)packing of ``.SerumPreset`` / ``.XferArpBank`` files.

Serum 2's own container format (undocumented by Xfer, reverse-engineered by the
community -- see ``docs/PARAMETER_SCHEMA.md``) is:

1. An 9-byte magic header: ``b"XferJson\\x00"``.
2. A little-endian ``(uint32 length, uint32 flags)`` pair, followed by that many
   bytes of UTF-8 JSON: the preset *metadata* (name, author, tags, product version...).
3. A second little-endian ``(uint32 length, uint32 flags)`` pair -- ``length`` is the
   size of the *uncompressed* payload, ``flags`` is a format marker (``2`` in every
   preset we've observed) -- followed by a Zstandard frame. Decompressing it yields
   CBOR-encoded bytes; decoding those CBOR bytes gives the actual synth engine state
   (oscillators, filters, envelopes, mod matrix, effects...).

This module only handles the container. It knows nothing about what the CBOR
*payload* means -- see :mod:`serum_mcp.preset.schema` for that.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cbor2
import zstandard

MAGIC = b"XferJson\x00"
_HEADER_STRUCT = struct.Struct("<II")  # (length, flags)
_ZSTD_COMPRESSION_LEVEL = 19
PAYLOAD_FLAGS = 2  # the format marker every observed preset carries


class PresetFormatError(ValueError):
    """Raised when a file does not look like a valid Serum 2 container."""


@dataclass
class SerumPreset:
    """A fully decoded Serum 2 preset: human-readable metadata + engine state."""

    metadata: dict[str, Any]
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return self.metadata.get("presetName", "")


@dataclass
class Container:
    """The undecoded parts of a container file, as stored on disk."""

    meta_flags: int
    meta_bytes: bytes
    payload_len: int
    payload_flags: int
    frame: bytes  # the zstd frame holding the CBOR payload

    def payload(self) -> bytes:
        cbor_bytes = zstandard.ZstdDecompressor().decompress(self.frame)
        if len(cbor_bytes) != self.payload_len:
            raise PresetFormatError(
                f"decompressed payload size mismatch: header says {self.payload_len}, "
                f"got {len(cbor_bytes)}"
            )
        return cbor_bytes


def split_container(raw: bytes) -> Container:
    """Split a ``.SerumPreset``/``.XferArpBank`` file into its stored parts."""
    if raw[: len(MAGIC)] != MAGIC:
        raise PresetFormatError(
            f"not a Serum 2 preset file: expected magic {MAGIC!r}, got {raw[: len(MAGIC)]!r}"
        )
    offset = len(MAGIC)

    meta_len, meta_flags = _HEADER_STRUCT.unpack_from(raw, offset)
    offset += _HEADER_STRUCT.size
    meta_bytes = raw[offset : offset + meta_len]
    offset += meta_len

    payload_len, payload_flags = _HEADER_STRUCT.unpack_from(raw, offset)
    offset += _HEADER_STRUCT.size
    return Container(meta_flags, meta_bytes, payload_len, payload_flags, raw[offset:])


def join_container(container: Container) -> bytes:
    """The inverse of :func:`split_container`."""
    return (
        MAGIC
        + _HEADER_STRUCT.pack(len(container.meta_bytes), container.meta_flags)
        + container.meta_bytes
        + _HEADER_STRUCT.pack(container.payload_len, container.payload_flags)
        + container.frame
    )


def compress_payload(cbor_bytes: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=_ZSTD_COMPRESSION_LEVEL).compress(cbor_bytes)


def unpack_bytes(raw: bytes) -> SerumPreset:
    """Decode the raw bytes of a ``.SerumPreset``/``.XferArpBank`` file."""
    container = split_container(raw)
    metadata = json.loads(container.meta_bytes)
    data = cbor2.loads(container.payload())
    return SerumPreset(metadata=metadata, data=data)


def unpack_file(path: str | Path) -> SerumPreset:
    """Decode a ``.SerumPreset``/``.XferArpBank`` file from disk."""
    return unpack_bytes(Path(path).read_bytes())


def pack_bytes(preset: SerumPreset) -> bytes:
    """Encode a :class:`SerumPreset` back into the on-disk container format."""
    meta_bytes = json.dumps(preset.metadata, separators=(",", ":")).encode("utf-8")
    cbor_bytes = cbor2.dumps(preset.data)
    return join_container(
        Container(
            meta_flags=0,
            meta_bytes=meta_bytes,
            payload_len=len(cbor_bytes),
            payload_flags=PAYLOAD_FLAGS,
            frame=compress_payload(cbor_bytes),
        )
    )


def pack_file(preset: SerumPreset, path: str | Path) -> Path:
    """Encode a :class:`SerumPreset` and write it to disk. Returns the written path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pack_bytes(preset))
    return dest
