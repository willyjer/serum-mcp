"""Lossless raw read and patch of ``.SerumPreset`` files.

:mod:`serum_mcp.preset.packer` round-trips the *decoded* data, but not the
bytes: ``cbor2`` re-encodes Serum's 32-bit floats as 64-bit ones, ``json``
escapes non-ASCII metadata, and Serum's own zstd output can't be reproduced at
any compression level. That's fine for presets serum-mcp generates, but a
patch to a real preset should change exactly the keys it names and nothing
else, down to the byte.

So this module carries its own small CBOR codec. It covers exactly the CBOR
that Serum writes (definite-length maps, arrays, strings and byte strings,
ints, floats, true/false/null; no tags) and remembers each float's width:
32-bit floats decode to ``float``, 64-bit floats to :class:`Float64`. Encoding
writes ``float`` as 32-bit, the way Serum does, and :class:`Float64` as
64-bit. A patched value is always written as 32-bit, even over a 64-bit
one. :class:`RawPreset` also keeps the original metadata and compressed
bytes, and reuses them whenever the re-encoded content is unchanged.

Paths are dotted, e.g. ``"Oscillator0.plainParams.kParamVolume"``; a segment
indexes a list when the container is a list. Keys that contain a dot (sample
map entries, for instance) can be read as part of their parent but not
addressed directly.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serum_mcp.preset.packer import (
    Container,
    PresetFormatError,
    compress_payload,
    join_container,
    split_container,
)

_DEFAULT_SENTINEL = "default"


class Float64(float):
    """A float that was stored (and will be re-stored) as a 64-bit CBOR float."""

    __slots__ = ()


class RawPathError(KeyError):
    """A path doesn't lead anywhere in the preset."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return str(self.args[0])


class RawValueError(ValueError):
    """A patch value can't be written where its path points."""


# --- CBOR codec ------------------------------------------------------------


def decode_cbor(data: bytes) -> Any:
    """Decode one CBOR item, keeping each float's width (see :class:`Float64`)."""
    value, offset = _decode(data, 0)
    if offset != len(data):
        raise PresetFormatError(f"{len(data) - offset} trailing bytes after CBOR payload")
    return value


def _decode_length(data: bytes, offset: int, info: int) -> tuple[int, int]:
    if info < 24:
        return info, offset
    size = {24: 1, 25: 2, 26: 4, 27: 8}.get(info)
    if size is None:
        raise PresetFormatError(f"unsupported CBOR length encoding {info} at byte {offset - 1}")
    return int.from_bytes(data[offset : offset + size], "big"), offset + size


def _decode(data: bytes, offset: int) -> tuple[Any, int]:
    initial = data[offset]
    major, info = initial >> 5, initial & 0x1F
    offset += 1
    if major == 7:
        if info == 20:
            return False, offset
        if info == 21:
            return True, offset
        if info == 22:
            return None, offset
        if info == 26:
            return struct.unpack_from(">f", data, offset)[0], offset + 4
        if info == 27:
            return Float64(struct.unpack_from(">d", data, offset)[0]), offset + 8
        raise PresetFormatError(f"unsupported CBOR simple value {info} at byte {offset - 1}")

    n, offset = _decode_length(data, offset, info)
    if major == 0:
        return n, offset
    if major == 1:
        return -1 - n, offset
    if major == 2:
        return bytes(data[offset : offset + n]), offset + n
    if major == 3:
        return data[offset : offset + n].decode("utf-8"), offset + n
    if major == 4:
        items = []
        for _ in range(n):
            item, offset = _decode(data, offset)
            items.append(item)
        return items, offset
    if major == 5:
        mapping = {}
        for _ in range(n):
            key, offset = _decode(data, offset)
            mapping[key], offset = _decode(data, offset)
        return mapping, offset
    raise PresetFormatError(f"unsupported CBOR major type {major} (tag) at byte {offset - 1}")


def encode_cbor(value: Any) -> bytes:
    """Encode ``value`` the way Serum does; the inverse of :func:`decode_cbor`."""
    out = bytearray()
    _encode(value, out)
    return bytes(out)


def _head(major: int, n: int) -> bytes:
    if n < 24:
        return bytes([major << 5 | n])
    if n < 1 << 8:
        return bytes([major << 5 | 24, n])
    if n < 1 << 16:
        return bytes([major << 5 | 25]) + n.to_bytes(2, "big")
    if n < 1 << 32:
        return bytes([major << 5 | 26]) + n.to_bytes(4, "big")
    return bytes([major << 5 | 27]) + n.to_bytes(8, "big")


def _encode(value: Any, out: bytearray) -> None:
    if value is False:
        out += b"\xf4"
    elif value is True:
        out += b"\xf5"
    elif value is None:
        out += b"\xf6"
    elif isinstance(value, Float64):
        out += b"\xfb" + struct.pack(">d", value)
    elif isinstance(value, float):
        out += b"\xfa" + struct.pack(">f", value)
    elif isinstance(value, int):
        out += _head(0, value) if value >= 0 else _head(1, -1 - value)
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        out += _head(3, len(encoded)) + encoded
    elif isinstance(value, bytes):
        out += _head(2, len(value)) + value
    elif isinstance(value, list):
        out += _head(4, len(value))
        for item in value:
            _encode(item, out)
    elif isinstance(value, dict):
        out += _head(5, len(value))
        for key, item in value.items():
            _encode(key, out)
            _encode(item, out)
    else:
        raise TypeError(f"can't CBOR-encode {type(value).__name__}")


# --- the container ---------------------------------------------------------


def _encode_metadata(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class Change:
    """One key a patch wrote. ``old`` is ``None`` when the key was ``created``."""

    path: str
    old: Any
    new: Any
    created: bool = False


@dataclass
class RawPreset:
    """A preset decoded losslessly, plus the stored container it came from."""

    metadata: dict[str, Any]
    data: dict[str, Any]
    _original: Container = dataclasses.field(repr=False)
    _original_cbor: bytes = dataclasses.field(repr=False)

    @classmethod
    def from_bytes(cls, raw: bytes) -> RawPreset:
        container = split_container(raw)
        cbor_bytes = container.payload()
        return cls(
            metadata=json.loads(container.meta_bytes),
            data=decode_cbor(cbor_bytes),
            _original=container,
            _original_cbor=cbor_bytes,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> RawPreset:
        return cls.from_bytes(Path(path).read_bytes())

    def encoded_cbor(self) -> bytes:
        """The current data, CBOR-encoded."""
        return encode_cbor(self.data)

    def to_bytes(self) -> bytes:
        """Encode the preset. Unchanged parts reuse the bytes they were read from."""
        original = self._original
        meta_bytes = _encode_metadata(self.metadata)
        if meta_bytes == _encode_metadata(json.loads(original.meta_bytes)):
            meta_bytes = original.meta_bytes
        cbor_bytes = self.encoded_cbor()
        if cbor_bytes == self._original_cbor:
            frame = original.frame
        else:
            frame = compress_payload(cbor_bytes)
        return join_container(
            dataclasses.replace(
                original, meta_bytes=meta_bytes, payload_len=len(cbor_bytes), frame=frame
            )
        )

    def write(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.to_bytes())
        return dest

    def patched(self, patches: dict[str, Any]) -> tuple[RawPreset, list[Change]]:
        """Return a copy with each ``path -> value`` in ``patches`` written, and
        what changed. Nothing else in the preset is touched.

        Numbers and booleans are written as 32-bit floats (booleans as 1.0/0.0),
        as Serum stores parameters, except where the existing value is an int or
        a bool (structural fields such as ``numFrames``), whose type is kept.
        Strings are written only over strings (enum values such as
        ``kParamType``). A ``plainParams: "default"`` sentinel becomes a dict
        holding just the patched key. A missing final key is created; a missing
        section is an error.
        """
        data = copy.deepcopy(self.data)
        changes = [_set_path(data, path, value) for path, value in patches.items()]
        return dataclasses.replace(self, metadata=copy.deepcopy(self.metadata), data=data), changes


# --- paths -----------------------------------------------------------------


def _split(path: str) -> list[str]:
    segments = path.split(".")
    if not path or any(not s for s in segments):
        raise RawPathError(f"invalid path {path!r}")
    return segments


def _child(container: Any, segment: str, walked: str) -> Any:
    where = walked or "the preset root"
    if isinstance(container, dict):
        if segment not in container:
            keys = ", ".join(map(str, list(container)[:30]))
            more = " ..." if len(container) > 30 else ""
            raise RawPathError(f"no key {segment!r} in {where} (keys: {keys}{more})")
        return container[segment]
    if isinstance(container, list):
        if not segment.isdigit() or int(segment) >= len(container):
            raise RawPathError(f"{where} is a list of {len(container)}; no index {segment!r}")
        return container[int(segment)]
    if container == _DEFAULT_SENTINEL:
        raise RawPathError(f"{where} is {_DEFAULT_SENTINEL!r} (all at Serum defaults)")
    raise RawPathError(f"{where} is a {type(container).__name__}, not a section")


def _walk(node: Any, segments: list[str], create_plain_params: bool = False) -> tuple[Any, str]:
    """Follow ``segments`` from ``node``; return where they lead and the path walked."""
    walked = ""
    for segment in segments:
        if create_plain_params and _is_default_plain_params(node, segment):
            node[segment] = {}
        node = _child(node, segment, walked)
        walked = f"{walked}.{segment}" if walked else segment
    return node, walked


def get_path(data: Any, path: str | None) -> Any:
    """Return the value at ``path``, or ``data`` itself when ``path`` is None."""
    if path is None:
        return data
    return _walk(data, _split(path))[0]


def _coerce(value: Any, existing: Any, path: str) -> Any:
    if isinstance(value, bool | int | float):
        if isinstance(existing, str):
            raise RawValueError(f"{path} holds the string {existing!r}; write a string")
        if isinstance(existing, dict | list):
            raise RawValueError(f"{path} is a section, not a value")
        if isinstance(existing, bool):
            if not isinstance(value, bool):
                raise RawValueError(f"{path} holds a bool; write true or false")
            return value
        if not math.isfinite(value):
            raise RawValueError(f"{path}: {value!r} isn't a finite number")
        if isinstance(existing, int):
            if isinstance(value, bool) or value != int(value):
                raise RawValueError(f"{path} holds an int; {value!r} isn't one")
            return int(value)
        try:
            return struct.unpack(">f", struct.pack(">f", float(value)))[0]
        except OverflowError:
            raise RawValueError(f"{path}: {value!r} is too large for a 32-bit float") from None
    if isinstance(value, str):
        if existing is not None and not isinstance(existing, str):
            raise RawValueError(f"{path} holds a {type(existing).__name__}; can't write a string")
        return value
    raise RawValueError(f"{path}: only numbers, booleans and strings can be patched")


def _set_path(data: Any, path: str, value: Any) -> Change:
    *parents, leaf = _split(path)
    node, walked = _walk(data, parents, create_plain_params=True)

    if isinstance(node, list):
        existing = _child(node, leaf, walked)
        node[int(leaf)] = new = _coerce(value, existing, path)
        return Change(path, existing, new)
    if not isinstance(node, dict):
        _child(node, leaf, walked)  # raises with a description of ``node``
    created = leaf not in node
    existing = None if created else node[leaf]
    if _is_default_plain_params(node, leaf):
        raise RawValueError(f"{path} is a section, not a value")
    node[leaf] = new = _coerce(value, existing, path)
    return Change(path, existing, new, created=created)


def _is_default_plain_params(node: Any, segment: str) -> bool:
    return (
        isinstance(node, dict)
        and segment == "plainParams"
        and node.get(segment) == _DEFAULT_SENTINEL
    )
