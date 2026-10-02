"""describe_preset shows every active mod route, including the ones
edit_preset can't model (pitch, FX internals, LFO point-mod buses), and
names tempo-synced LFO rates instead of printing ``rate=0``."""

from __future__ import annotations

from pathlib import Path

from serum_mcp.preset.packer import pack_file, unpack_file
from serum_mcp.tools.describe_preset import describe_preset

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _describe(tmp_path, patch) -> str:
    """describe_preset on the init fixture with ``patch(data)`` applied."""
    preset = unpack_file(_FIXTURES / "init_preset.SerumPreset")
    patch(preset.data)
    return describe_preset(str(pack_file(preset, tmp_path / "patched.SerumPreset")))


def _route(dest_type: str, dest_id: int, param: str, source: list[int], amount: float) -> dict:
    # destModuleParamID is irrelevant to describe; real slots carry one.
    return {
        "destModuleID": dest_id,
        "destModuleParamID": 0,
        "destModuleParamName": param,
        "destModuleTypeString": dest_type,
        "plainParams": {"kParamAmount": amount},
        "source": source,
    }


def _mod_matrix(summary: str) -> list[str]:
    lines = summary.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Mod matrix"))
    return [line for line in lines[start + 1 :] if line.startswith("  - ")]


def test_shows_pitch_routes_edit_preset_cannot_model(tmp_path):
    # Shaped like Factory "ARP - Bell Arperium" ModSlot2: macro0 -> Osc B coarse pitch.
    def patch(data):
        data["ModSlot0"] = _route("Oscillator", 0, "kParamVolume", [6, 0], 50.0)
        data["ModSlot1"] = _route("Oscillator", 1, "kParamCoarsePit", [25, 0], -23.5885)
        data["ModSlot2"] = _route("Global", 0, "kParamMasterTuning", [7, 0], 12.0)
        data["ModSlot3"] = {"plainParams": "default"}

    summary = _describe(tmp_path, patch)

    routes = _mod_matrix(summary)
    assert len(routes) == 3
    assert any("lfo0 -> oscillator0.volume: +50%" in r for r in routes)
    assert any("macro0 -> Oscillator1.kParamCoarsePit: -24%" in r for r in routes)
    assert any("lfo1 -> Global0.kParamMasterTuning: +12%" in r for r in routes)
    assert "2 shown by raw name" in summary


def test_names_fx_destinations_by_rack_and_position(tmp_path):
    # FX destinations use destModuleID = rack * 100 + position in that rack.
    def patch(data):
        data["ModSlot0"] = _route("FXFilter", 101, "kParamFreq", [6, 0], 40.0)

    routes = _mod_matrix(_describe(tmp_path, patch))

    assert routes == ["  - lfo0 -> FXRack1[1].FXFilter.kParamFreq: +40%  (raw)"]


def test_shows_routes_from_undecoded_sources_by_id(tmp_path):
    def patch(data):
        data["ModSlot0"] = _route("Oscillator", 0, "kParamVolume", [999, 0], 10.0)

    routes = _mod_matrix(_describe(tmp_path, patch))

    assert routes == ["  - source#999 -> oscillator0.volume: +10%  (raw)"]


def test_ignores_slots_without_a_destination(tmp_path):
    # 12 real slots carry a source but no destination; Serum shows nothing for them.
    def patch(data):
        data["ModSlot0"] = {"plainParams": {"kParamAmount": 50.0}, "source": [31, 0]}

    assert "Mod matrix: (no active routes)" in _describe(tmp_path, patch)


def test_shows_a_synced_lfo_that_drives_a_route(tmp_path):
    # An untouched LFO is synced at 1/4 and stores no rate at all; it used to
    # be left out of the LFO list and print rate=0 when shown.
    def patch(data):
        data["ModSlot0"] = _route("Oscillator", 0, "kParamCoarsePit", [6, 0], 24.0)

    summary = _describe(tmp_path, patch)

    assert "LFO 1: rate=sync 1/4" in summary


def test_shows_unknown_synced_divisions_as_raw_rates(tmp_path):
    def patch(data):
        data["LFO0"] = {"plainParams": {"kParamRate": 10.66}}
        data["LFO1"] = {"plainParams": {"kParamRate": 37.5, "kParamTriplets": 1.0}}
        data["LFO2"] = {"plainParams": {"kParamRate": 4.0, "kParamBeatSync": 0.0}}

    summary = _describe(tmp_path, patch)

    assert "LFO 1: rate=sync 1/8" in summary
    assert "LFO 2: rate=sync raw 37.50 triplets" in summary
    assert "LFO 3: rate=4.00Hz" in summary


def test_tags_a_route_with_an_undecoded_aux_source_as_raw(tmp_path):
    # extract_spec drops an aux source it can't name, so edit_preset can't
    # address this route as it stands.
    def patch(data):
        data["ModSlot0"] = _route("Oscillator", 0, "kParamVolume", [6, 999], 10.0)

    routes = _mod_matrix(_describe(tmp_path, patch))

    assert routes == ["  - lfo0 via source#999 -> oscillator0.volume: +10%  (raw)"]


def test_shows_a_free_lfo_without_a_stored_rate_at_the_measured_default(tmp_path):
    def patch(data):
        data["LFO0"] = {"plainParams": {"kParamBeatSync": 0.0, "kParamMode": "kLFOMode_Trig"}}

    assert "LFO 1: rate=6.25Hz (default)" in _describe(tmp_path, patch)
