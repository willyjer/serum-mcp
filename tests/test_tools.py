"""Tests for the MCP tool functions. There's no LLM call anywhere in this
package to mock out -- generate_preset/edit_preset take a structured
PresetSpec directly (built by the calling model, e.g. Claude Code itself),
so these tests just exercise the deterministic validate/merge/pack path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from serum_mcp import config
from serum_mcp.generation.spec import EnvelopeSpec, FilterSpec, OscillatorSpec, PresetSpec
from serum_mcp.preset.packer import pack_file, unpack_file
from serum_mcp.tools import describe_preset as describe_preset_mod
from serum_mcp.tools import edit_preset as edit_preset_mod
from serum_mcp.tools import generate_preset as generate_preset_mod
from serum_mcp.tools.list_parameters import list_parameters


def _bass_spec(**overrides) -> PresetSpec:
    defaults = dict(
        name="BA - Simple LP Bass",
        description="Simple bass with a warm low-pass filtered oscillator.",
        oscillators=[OscillatorSpec(enabled=True, octave=-1, volume=0.8)],
        filters=[FilterSpec(enabled=True, type="lowpass_24", cutoff=0.35, resonance=15)],
        envelopes=[EnvelopeSpec(attack=0.001, decay=0.3, sustain=0.6, release=0.2)],
    )
    defaults.update(overrides)
    return PresetSpec(**defaults)


@pytest.fixture
def presets_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path))
    return tmp_path


def test_generate_preset_writes_valid_file(presets_dir):
    path = generate_preset_mod.generate_preset(_bass_spec())

    written = Path(path)
    assert written.exists()
    assert written.parent == presets_dir
    preset = unpack_file(written)
    assert preset.metadata["presetName"] == "BA - Simple LP Bass"
    assert preset.data["VoiceFilter0"]["plainParams"]["kParamType"] == "L24"


def test_generate_preset_with_subfolder_nests_the_file(presets_dir):
    path = generate_preset_mod.generate_preset(_bass_spec(), subfolder="RAGE Bank")

    written = Path(path)
    assert written.exists()
    assert written.parent == presets_dir / "RAGE Bank"


def test_generate_preset_with_nested_subfolder(presets_dir):
    path = generate_preset_mod.generate_preset(_bass_spec(), subfolder="RAGE Bank/Leads")

    written = Path(path)
    assert written.parent == presets_dir / "RAGE Bank" / "Leads"


def test_generate_preset_subfolder_strips_path_traversal(presets_dir):
    path = generate_preset_mod.generate_preset(_bass_spec(), subfolder="../../etc/../../RAGE Bank")

    written = Path(path)
    # Traversal segments collapse away entirely; only the real folder name
    # survives, and the file must still land inside the presets folder.
    assert presets_dir in written.parents
    assert written.parent == presets_dir / "etc" / "RAGE Bank"


def test_generate_preset_without_subfolder_still_writes_flat(presets_dir):
    path = generate_preset_mod.generate_preset(_bass_spec(), subfolder=None)

    written = Path(path)
    assert written.parent == presets_dir


def test_edit_preset_without_rename_keeps_same_path(presets_dir):
    path = generate_preset_mod.generate_preset(_bass_spec())

    # A partial spec: only the filter changes, name unchanged.
    edit_spec = PresetSpec(
        name="BA - Simple LP Bass",
        description="",
        filters=[FilterSpec(enabled=True, type="lowpass_24", cutoff=0.9, resonance=15)],
    )
    edited_path = edit_preset_mod.edit_preset(path, edit_spec)

    assert edited_path == path
    preset = unpack_file(edited_path)
    assert preset.data["VoiceFilter0"]["plainParams"]["kParamFreq"] == 0.9
    # Untouched by the edit spec (empty oscillators list):
    assert preset.data["Oscillator0"]["plainParams"]["kParamOctave"] == -1.0


def test_edit_preset_renames_file_when_name_changes(presets_dir):
    """Regression test: Serum's preset browser displays the filename, not
    the internal presetName metadata (confirmed against a live Serum 2
    install -- editing metadata alone left the name shown in Serum
    unchanged). edit_preset must rename the file to match a changed name,
    not just update metadata, and must clean up the old file."""
    path = generate_preset_mod.generate_preset(_bass_spec())

    edit_spec = PresetSpec(
        name="BA - Simple LP Bass (brighter)",
        description="",
        filters=[FilterSpec(enabled=True, type="lowpass_24", cutoff=0.9, resonance=15)],
    )
    edited_path = edit_preset_mod.edit_preset(path, edit_spec)

    assert edited_path != path
    assert Path(edited_path).name == "BA - Simple LP Bass brighter.SerumPreset"
    assert not Path(path).exists()  # old file removed
    preset = unpack_file(edited_path)
    assert preset.data["VoiceFilter0"]["plainParams"]["kParamFreq"] == 0.9
    assert preset.metadata["presetName"] == "BA - Simple LP Bass (brighter)"
    # Untouched by the edit spec (empty oscillators list):
    assert preset.data["Oscillator0"]["plainParams"]["kParamOctave"] == -1.0


def test_generate_preset_plain_spec_has_no_dependency_note(presets_dir):
    """A preset using only curated wavetables/multisample instruments is
    fully self-contained (portable to another machine as just the one
    .SerumPreset file) -- the return value must be exactly the path, no
    extra lines, matching every pre-existing caller's ``Path(result)``
    usage."""
    result = generate_preset_mod.generate_preset(_bass_spec())

    assert "\n" not in result
    assert Path(result).exists()


def test_generate_preset_custom_harmonics_surfaces_dependency_note(
    presets_dir, tmp_path, monkeypatch
):
    """custom_harmonics writes a real .wav to the local Tables folder that
    the .SerumPreset only references by relative path -- Serum's own
    limitation, not embedded content. The tool's return value must call
    this out (first line still the real path) rather than silently
    producing a preset that breaks on another machine."""
    monkeypatch.setenv(config.TABLES_ENV_VAR, str(tmp_path))
    spec = _bass_spec(
        oscillators=[OscillatorSpec(enabled=True, custom_harmonics=[[1.0, 0.5, 0.25]], octave=-1)]
    )

    result = generate_preset_mod.generate_preset(spec)

    lines = result.splitlines()
    assert len(lines) > 1
    written = Path(lines[0])
    assert written.exists()
    assert "depends on 1 local file" in result
    dependency_line = next(line for line in lines if line.strip().startswith("- "))
    dependency_path = Path(dependency_line.strip()[2:])
    assert dependency_path.exists()
    assert dependency_path.suffix == ".wav"


def test_describe_preset_mentions_key_sections():
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    summary = describe_preset_mod.describe_preset(str(fixtures_dir / "init_preset.SerumPreset"))
    assert "Osc A" in summary
    assert "Filter 1" in summary
    assert "Env 1" in summary


def _init_preset_with(tmp_path, patch) -> str:
    """The init fixture with ``patch(data)`` applied, written to tmp_path."""
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    preset = unpack_file(fixtures_dir / "init_preset.SerumPreset")
    patch(preset.data)
    return str(pack_file(preset, tmp_path / "patched.SerumPreset"))


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        # Factory "SEQ - A Hint of Retro": a balance leaning to Filter 1.
        (
            lambda d: d.__setitem__(
                "RoutingSlot0", {"plainParams": {"kParamFilterBalance": -98.59649119898677}}
            ),
            "Osc A",
        ),
        # "BRAINWAVEZ - BS Guilty": a 32 s hold.
        (lambda d: d.__setitem__("Env0", {"plainParams": {"kParamHold": 32.0}}), "hold=32000.0ms"),
        # Factory "SC - Birds": a spectral window starting below 20 Hz.
        (
            lambda d: (
                d["Oscillator0"].__setitem__("plainParams", {"kParamType": "kOsc_Spectral"}),
                d["Oscillator0"]["SpectralOsc0"].__setitem__(
                    "plainParams", {"kParamFreqLo": 15.945181234369928}
                ),
            ),
            "Osc A",
        ),
    ],
    ids=["negative-filter-balance", "32s-hold", "spectral-freq-lo-below-20"],
)
def test_describe_preset_reads_real_values_that_used_to_crash_it(tmp_path, patch, expected):
    summary = describe_preset_mod.describe_preset(_init_preset_with(tmp_path, patch))
    assert expected in summary


def test_list_parameters_is_valid_json_with_expected_sections():
    import json

    parsed = json.loads(list_parameters())
    assert "oscillator" in parsed
    assert "voice_filter" in parsed
    assert "fx_params" in parsed
    assert "mod_source_ids" in parsed
    assert "mod_dest_targets" in parsed
    assert "kParamFreq" in parsed["voice_filter"]

    roles = parsed["role_starting_points"]
    assert set(roles) == {
        "bass",
        "pluck",
        "lead",
        "pad",
        "chords",
        "synth",
        "arp",
        "sequence",
    }
    assert roles["bass"]["mono"] is True
    assert roles["pluck"]["envelope"]["sustain"] == 0.0
