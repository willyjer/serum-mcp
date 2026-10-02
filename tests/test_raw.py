"""Tests for the lossless raw read/patch layer.

Real Serum 2 presets store every float as a 32-bit CBOR float (``fa``), while
presets written through ``cbor2`` (like ``fixtures/init_preset.SerumPreset``)
store 64-bit floats (``fb``). The raw layer must reproduce either exactly, so
these tests check byte-for-byte equality, not just decoded equality.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest
import zstandard

from serum_mcp.preset.packer import MAGIC
from serum_mcp.preset.raw import (
    Float64,
    RawPathError,
    RawPreset,
    RawValueError,
    decode_cbor,
    encode_cbor,
    get_path,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
INIT_FIXTURE = FIXTURES_DIR / "init_preset.SerumPreset"


def _f32(v: float) -> bytes:
    return b"\xfa" + struct.pack(">f", v)


def _f64(v: float) -> bytes:
    return b"\xfb" + struct.pack(">d", v)


def _text(s: str) -> bytes:
    b = s.encode()
    assert len(b) < 24
    return bytes([0x60 | len(b)]) + b


def _container(metadata: bytes, cbor: bytes, level: int = 7) -> bytes:
    frame = zstandard.ZstdCompressor(level=level).compress(cbor)
    return (
        MAGIC
        + struct.pack("<II", len(metadata), 0)
        + metadata
        + struct.pack("<II", len(cbor), 2)
        + frame
    )


def _flatten(value, prefix=()):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(_flatten(v, (*prefix, k)))
        return out
    if isinstance(value, list):
        out = {}
        for i, v in enumerate(value):
            out.update(_flatten(v, (*prefix, i)))
        return out
    return {prefix: (type(value), value)}


# A Serum-style payload: 32-bit floats, a "default" plainParams sentinel, an
# int structural field, a list, a bool, null, and a sparse plainParams dict.
SERUM_STYLE_CBOR = (
    b"\xa3"
    + _text("Oscillator0")
    + b"\xa2"
    + _text("plainParams")
    + b"\xa2"
    + _text("kParamVolume")
    + _f32(0.75)
    + _text("kParamType")
    + _text("kOsc_WT")
    + _text("WTOsc0")
    + b"\xa2"
    + _text("numFrames")
    + b"\x19\x48\x00"  # 18432
    + _text("plainParams")
    + _text("default")
    + _text("Env0")
    + b"\xa1"
    + _text("plainParams")
    + _text("default")
    + _text("misc")
    + b"\x85"
    + _f32(0.30000001192092896)
    + b"\xf5"
    + b"\xf6"
    + b"\x38\x63"  # -100
    + b"\x43abc"
)
SERUM_STYLE_META = '{"fileType":"SerumPreset","presetName":"Größe ❄️"}'.encode()


# --- codec ---------------------------------------------------------------


@pytest.mark.parametrize(
    "cbor",
    [
        SERUM_STYLE_CBOR,
        _f64(0.1),
        _f32(1.0),
        b"\x82" + _f32(0.5) + _f64(0.5),
        b"\x1b" + struct.pack(">Q", 2**40),
        b"\x1a" + struct.pack(">I", 70000),
        b"\x3a" + struct.pack(">I", 70000),
        b"\xf4",
        b"\x40",
        b"\xa0",
    ],
)
def test_codec_round_trip_is_byte_identical(cbor):
    assert encode_cbor(decode_cbor(cbor)) == cbor


def test_codec_keeps_float_width():
    decoded = decode_cbor(b"\x82" + _f32(0.5) + _f64(0.5))
    assert type(decoded[0]) is float
    assert type(decoded[1]) is Float64


def test_codec_rejects_trailing_bytes():
    with pytest.raises(ValueError):
        decode_cbor(_f32(1.0) + b"\x00")


def test_init_fixture_file_round_trip_is_byte_identical():
    raw = INIT_FIXTURE.read_bytes()
    assert RawPreset.from_bytes(raw).to_bytes() == raw


def test_serum_style_file_round_trip_is_byte_identical():
    # Compressed at a level serum-mcp never uses: the original frame must be
    # reused, since Serum's own zstd output can't be reproduced.
    raw = _container(SERUM_STYLE_META, SERUM_STYLE_CBOR, level=7)
    preset = RawPreset.from_bytes(raw)
    assert preset.metadata["presetName"] == "Größe ❄️"
    assert preset.to_bytes() == raw


def test_patch_nothing_is_byte_identical():
    raw = _container(SERUM_STYLE_META, SERUM_STYLE_CBOR)
    patched, changes = RawPreset.from_bytes(raw).patched({})
    assert changes == []
    assert patched.to_bytes() == raw

    init = INIT_FIXTURE.read_bytes()
    assert RawPreset.from_bytes(init).patched({})[0].to_bytes() == init


# --- reading -------------------------------------------------------------


def test_get_path_returns_subtree_or_whole():
    data = decode_cbor(SERUM_STYLE_CBOR)
    assert get_path(data, None) is data
    assert get_path(data, "Oscillator0.plainParams.kParamVolume") == 0.75
    assert get_path(data, "Oscillator0.WTOsc0")["numFrames"] == 18432
    assert get_path(data, "misc.3") == -100


@pytest.mark.parametrize(
    "path",
    ["Nope", "Oscillator0.plainParams.kParamNope", "Env0.plainParams.kParamAttack", "misc.9"],
)
def test_get_path_missing_raises(path):
    with pytest.raises(RawPathError):
        get_path(decode_cbor(SERUM_STYLE_CBOR), path)


# --- patching ------------------------------------------------------------


def _serum_style() -> RawPreset:
    return RawPreset.from_bytes(_container(SERUM_STYLE_META, SERUM_STYLE_CBOR))


def test_patch_changes_only_named_keys():
    original = RawPreset.from_bytes(INIT_FIXTURE.read_bytes())
    patched, changes = original.patched(
        {"Env0.plainParams.kParamAttack": 0.25, "Oscillator0.plainParams.kParamVolume": 0.5}
    )
    again = RawPreset.from_bytes(patched.to_bytes())

    # Both sections start as the "default" sentinel, which becomes a dict
    # holding just the one patched key.
    before, after = _flatten(original.data), _flatten(again.data)
    assert before.keys() - after.keys() == {("Env0", "plainParams"), ("Oscillator0", "plainParams")}
    assert after.keys() - before.keys() == {
        ("Env0", "plainParams", "kParamAttack"),
        ("Oscillator0", "plainParams", "kParamVolume"),
    }
    # Every other value is unchanged, down to its exact type (float width).
    for key in before.keys() & after.keys():
        assert before[key] == after[key]
    assert {c.path for c in changes} == {
        "Env0.plainParams.kParamAttack",
        "Oscillator0.plainParams.kParamVolume",
    }
    assert again.metadata == original.metadata


def test_patch_does_not_mutate_original():
    original = _serum_style()
    snapshot = _flatten(original.data)
    original.patched({"Oscillator0.plainParams.kParamVolume": 0.1})
    assert _flatten(original.data) == snapshot


def test_patched_values_are_32_bit_floats():
    patched, _ = _serum_style().patched(
        {"Oscillator0.plainParams.kParamVolume": 0.3, "Env0.plainParams.kParamAttack": 2}
    )
    cbor = patched.encoded_cbor()
    assert _text("kParamVolume") + _f32(0.3) in cbor
    assert _text("kParamAttack") + _f32(2.0) in cbor
    assert b"\xfb" not in cbor


def test_patched_booleans_are_written_as_floats():
    patched, changes = _serum_style().patched(
        {"Env0.plainParams.kParamOn": True, "Oscillator0.plainParams.kParamVolume": False}
    )
    cbor = patched.encoded_cbor()
    assert _text("kParamOn") + _f32(1.0) in cbor
    assert _text("kParamVolume") + _f32(0.0) in cbor
    assert [c.new for c in changes] == [1.0, 0.0]


def test_patch_rounds_to_what_is_stored():
    patched, changes = _serum_style().patched({"Oscillator0.plainParams.kParamVolume": 0.3})
    stored = get_path(patched.data, "Oscillator0.plainParams.kParamVolume")
    assert stored == struct.unpack(">f", struct.pack(">f", 0.3))[0]
    assert changes[0].old == 0.75
    assert changes[0].new == stored


def test_patch_into_default_plain_params_creates_only_that_key():
    patched, changes = _serum_style().patched({"Env0.plainParams.kParamAttack": 0.5})
    assert patched.data["Env0"]["plainParams"] == {"kParamAttack": 0.5}
    assert changes[0].old is None and changes[0].created


def test_patch_keeps_int_fields_int():
    patched, _ = _serum_style().patched({"Oscillator0.WTOsc0.numFrames": 2048})
    value = get_path(patched.data, "Oscillator0.WTOsc0.numFrames")
    assert type(value) is int and value == 2048


def test_patch_list_element():
    patched, _ = _serum_style().patched({"misc.0": 0.5})
    assert patched.data["misc"][0] == 0.5


def test_patch_enum_string():
    patched, _ = _serum_style().patched({"Oscillator0.plainParams.kParamType": "kOsc_Sample"})
    assert patched.data["Oscillator0"]["plainParams"]["kParamType"] == "kOsc_Sample"


@pytest.mark.parametrize(
    ("patches", "error"),
    [
        ({"Nope.plainParams.kParamVolume": 1.0}, RawPathError),
        ({"misc.9": 1.0}, RawPathError),
        ({"Oscillator0.plainParams.kParamType": 1.0}, RawValueError),
        ({"Oscillator0.plainParams.kParamVolume": "loud"}, RawValueError),
        ({"Oscillator0.WTOsc0.numFrames": 1.5}, RawValueError),
        ({"Oscillator0.plainParams.kParamVolume": [1.0]}, RawValueError),
        ({"Oscillator0.plainParams": 1.0}, RawValueError),
        ({"": 1.0}, RawPathError),
        ({"Oscillator0.WTOsc0.numFrames": True}, RawValueError),
        ({"Oscillator0.WTOsc0.numFrames": float("inf")}, RawValueError),
        ({"Oscillator0.WTOsc0.numFrames": float("nan")}, RawValueError),
        ({"Oscillator0.plainParams.kParamVolume": 1e40}, RawValueError),
        ({"Oscillator0.plainParams.kParamVolume": float("nan")}, RawValueError),
    ],
)
def test_patch_rejects_bad_patches(patches, error):
    with pytest.raises(error):
        _serum_style().patched(patches)


# --- the local preset library (opt-in) -----------------------------------


@pytest.mark.skipif(
    not os.environ.get("SERUM_RAW_SURVEY"),
    reason="set SERUM_RAW_SURVEY=1 to round-trip every preset under SERUM_PRESETS_PATH's parent",
)
def test_local_library_round_trips_byte_identical():
    from serum_mcp.config import get_presets_dir

    root = get_presets_dir().parent
    files = list(root.rglob("*.SerumPreset"))
    assert files
    failures = [
        f for f in files if RawPreset.from_bytes(f.read_bytes()).to_bytes() != f.read_bytes()
    ]
    assert failures == []


def test_header_flags_and_unchanged_metadata_bytes_are_kept():
    meta = b'{"presetName":"X","n":1.0}'
    raw = _container(meta, SERUM_STYLE_CBOR)
    raw = raw[: len(MAGIC) + 4] + struct.pack("<I", 7) + raw[len(MAGIC) + 8 :]
    preset = RawPreset.from_bytes(raw)
    assert preset.to_bytes() == raw
    preset.metadata["n"] = 1  # equal to 1.0 in Python, but not the same JSON
    assert b'"n":1}' in preset.to_bytes()
