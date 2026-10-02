"""Tests for the read_raw / patch_raw MCP tool functions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from serum_mcp.preset.raw import RawPathError, RawPreset
from serum_mcp.tools.patch_raw import patch_raw
from serum_mcp.tools.read_raw import read_raw

INIT_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "init_preset.SerumPreset"


@pytest.fixture
def preset(tmp_path) -> Path:
    path = tmp_path / "Init.SerumPreset"
    shutil.copy(INIT_FIXTURE, path)
    return path


def test_read_raw_whole_preset(preset):
    out = json.loads(read_raw(str(preset)))
    assert out["metadata"]["presetName"] == "Init"
    assert out["data"]["Oscillator0"]["WTOsc0"]["numFrames"] == 18432


def test_read_raw_subtree(preset):
    out = json.loads(read_raw(str(preset), "Oscillator0.WTOsc0"))
    assert out == {
        "path": "Oscillator0.WTOsc0",
        "value": RawPreset.from_file(preset).data["Oscillator0"]["WTOsc0"],
    }


def test_read_raw_shows_32_bit_floats_at_their_shortest(preset, tmp_path):
    patched = tmp_path / "patched.SerumPreset"
    patch_raw(str(preset), {"Env0.plainParams.kParamAttack": 0.3}, str(patched))
    out = json.loads(read_raw(str(patched), "Env0.plainParams.kParamAttack"))
    assert out["value"] == 0.3


def test_read_raw_missing_path_raises(preset):
    with pytest.raises(RawPathError):
        read_raw(str(preset), "Oscillator0.Nope")


def test_patch_raw_writes_output_and_reports_changes(preset, tmp_path):
    source_bytes = preset.read_bytes()
    dest = tmp_path / "out" / "Patched.SerumPreset"
    result = patch_raw(
        str(preset),
        {"Env0.plainParams.kParamAttack": 0.5, "Oscillator0.WTOsc0.numFrames": 2048},
        str(dest),
    )
    lines = result.splitlines()
    assert lines[0] == str(dest.resolve())
    assert "Env0.plainParams.kParamAttack: (absent) -> 0.5" in lines
    assert "Oscillator0.WTOsc0.numFrames: 18432 -> 2048" in lines
    assert preset.read_bytes() == source_bytes
    written = RawPreset.from_file(dest).data
    assert written["Env0"]["plainParams"] == {"kParamAttack": 0.5}


def test_patch_raw_with_no_patches_copies_byte_identical(preset, tmp_path):
    dest = tmp_path / "copy.SerumPreset"
    result = patch_raw(str(preset), {}, str(dest))
    assert result.splitlines()[1] == "No changes."
    assert dest.read_bytes() == preset.read_bytes()


def test_patch_raw_writes_nothing_on_a_bad_patch(preset, tmp_path):
    dest = tmp_path / "bad.SerumPreset"
    with pytest.raises(RawPathError):
        patch_raw(str(preset), {"Env0.plainParams.kParamAttack": 0.5, "Nope.x": 1}, str(dest))
    assert not dest.exists()
