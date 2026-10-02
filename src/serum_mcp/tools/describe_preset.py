"""``describe_preset`` MCP tool implementation."""

from __future__ import annotations

from serum_mcp.generation.spec import LfoSpec
from serum_mcp.preset.introspect import active_mod_routes, count_unmodeled_fx_units, extract_spec
from serum_mcp.preset.packer import unpack_file

_OSC_LABELS = ("A", "B", "C", "Noise", "Sub")

# Synced kParamRate -> division. Only these two points are known (see
# schema.LFO_PARAMS["kParamRate"]): absent (read as 0.0) is 1/4, 10.66 is 1/8.
# Any other synced value is shown raw rather than guessed.
_KNOWN_SYNC_DIVISIONS = {0.0: "1/4", 10.66: "1/8"}


def _lfo_rate(lfo: LfoSpec) -> str:
    """An LFO is tempo-synced unless kParamBeatSync is explicitly off."""
    if lfo.beat_sync is False:
        return f"{lfo.rate:.2f}Hz" if lfo.rate else "free default"
    division = _KNOWN_SYNC_DIVISIONS.get(round(lfo.rate, 2))
    rate = f"sync {division}" if division else f"sync raw {lfo.rate:.2f}"
    dotted = " dotted" if lfo.dotted else ""
    triplets = " triplets" if lfo.triplets else ""
    return f"{rate}{dotted}{triplets}"


def describe_preset(preset_path: str) -> str:
    """Return a human-readable summary of an existing preset's sound-shaping
    parameters, so a user can understand what was generated/edited without
    opening Serum."""
    preset = unpack_file(preset_path)
    spec = extract_spec(preset.data)

    lines = [f"Preset: {preset.metadata.get('presetName') or '(unnamed)'}"]
    if preset.metadata.get("presetDescription"):
        lines.append(f"Description: {preset.metadata['presetDescription']}")
    lines.append(f"Author: {preset.metadata.get('presetAuthor', '?')}")
    lines.append("")

    for i, (label, osc) in enumerate(zip(_OSC_LABELS, spec.oscillators, strict=True)):
        state = "ON " if osc.enabled else "off"
        common = f"octave={osc.octave:+.0f}  volume={osc.volume:.2f}  pan={osc.pan:+.0f}"
        if i in (0, 1, 2):
            warp2 = f"  warp2={osc.warp_mode2}({osc.warp_amount2:.2f})" if osc.warp_mode2 else ""
            if osc.sample_playback_source:
                loop = (
                    f"  loop={osc.sample_loop}"
                    f"({osc.sample_loop_start:.0f}-{osc.sample_loop_end:.0f}%)"
                    if osc.sample_loop != "off"
                    else "  loop=off (one-shot)"
                )
                extra = (
                    f"sample={osc.sample_playback_source}  "
                    f"warp={osc.warp_mode}({osc.warp_amount:.2f}){warp2}{loop}"
                )
            elif osc.granular_source:
                extra = (
                    f"granular={osc.granular_source}  "
                    f"density={osc.granular_density:.1f}Hz  "
                    f"grain_len={osc.granular_grain_length:.1f}ms  "
                    f"warp={osc.warp_mode}({osc.warp_amount:.2f}){warp2}"
                )
            elif osc.spectral_source:
                extra = (
                    f"spectral={osc.spectral_source}  "
                    f"freq={osc.spectral_warp_freq_lo:.0f}-{osc.spectral_warp_freq_hi:.0f}Hz  "
                    f"warp={osc.warp_mode}({osc.warp_amount:.2f}){warp2}"
                )
            elif osc.multisample_source:
                extra = (
                    f"multisample={osc.multisample_source}  "
                    f"env(A/D/R)={osc.multisample_env_attack:.2f}/"
                    f"{osc.multisample_env_decay:.2f}/{osc.multisample_env_release:.2f}s  "
                    f"warp={osc.warp_mode}({osc.warp_amount:.2f}){warp2}"
                )
            else:
                extra = (
                    f"wavetable={osc.wavetable}  table_pos={osc.table_position:.1f}  "
                    f"warp={osc.warp_mode}({osc.warp_amount:.2f}){warp2}"
                )
            if osc.unison > 1:
                extra += f"  unison={osc.unison:.0f}  detune={osc.detune:.2f}"
        elif i == 3:
            extra = f"noise_type={osc.noise_type}"
        else:
            extra = f"sub_shape={osc.sub_shape}"
        lines.append(f"Osc {label}: {state}  {common}  {extra}")

    lines.append("")
    for i, flt in enumerate(spec.filters, start=1):
        state = "ON " if flt.enabled else "off"
        stereo = f"  stereo={flt.stereo:.0f}%" if flt.stereo else ""
        var = f"  var={flt.var:.0f}%" if flt.var else ""
        key_track = "  key_track" if flt.key_track else ""
        lines.append(
            f"Filter {i}: {state}  type={flt.type}  cutoff={flt.cutoff:.2f}  "
            f"resonance={flt.resonance:.0f}%  drive={flt.drive:.0f}%{stereo}{var}{key_track}"
        )

    lines.append("")
    for i, env in enumerate(spec.envelopes, start=1):
        hold = f"  hold={env.hold * 1000:.1f}ms" if env.hold else ""
        lines.append(
            f"Env {i}: attack={env.attack * 1000:.1f}ms{hold}  decay={env.decay:.2f}s  "
            f"sustain={env.sustain:.2f}  release={env.release:.2f}s"
        )

    routes = active_mod_routes(preset.data)
    route_sources = {r.source for r in routes} | {r.aux_source for r in routes}
    active_lfos = [
        (i, lfo)
        for i, lfo in enumerate(spec.lfos, start=1)
        if lfo.rate or lfo.mode != "Free" or lfo.shape or f"lfo{i - 1}" in route_sources
    ]
    if active_lfos:
        lines.append("")
        for i, lfo in active_lfos:
            delay = f"  delay={lfo.delay:.2f}s" if lfo.delay else ""
            shape = f"  shape={lfo.shape}" if lfo.shape else ""
            mono = "  mono" if lfo.mono else ""
            lines.append(f"LFO {i}: rate={_lfo_rate(lfo)}  mode={lfo.mode}{delay}{shape}{mono}")

    active_macros = [(i, m) for i, m in enumerate(spec.macros, start=1) if m.value or m.name]
    if active_macros:
        lines.append("")
        for i, macro in active_macros:
            name = f' "{macro.name}"' if macro.name else ""
            lines.append(f"Macro {i}{name}: {macro.value:.0f}%")

    lines.append("")
    if spec.fx_chain:
        lines.append("FX chain:")
        for fx in spec.fx_chain:
            lines.append(f"  - {fx.type} (wet={fx.wet:.0f}%)")
    else:
        lines.append("FX chain: (empty)")
    unmodeled_fx = count_unmodeled_fx_units(preset.data)
    if unmodeled_fx:
        lines.append(
            f"  (+ {unmodeled_fx} unit(s) in parallel/multiband FX routing -- "
            "not decoded, not shown above, but still present in the file)"
        )

    lines.append("")
    if routes:
        raw = sum(not r.modeled for r in routes)
        raw_note = (
            f"; {raw} shown by raw name, which edit_preset's mod_routes can't address"
            if raw
            else ""
        )
        lines.append(f"Mod matrix ({len(routes)} active routes{raw_note}):")
        for route in routes:
            aux = f" via {route.aux_source}" if route.aux_source else ""
            bip = ", bipolar" if route.bipolar else ""
            curve = ", curve" if route.curve else ""
            tag = "  (raw)" if not route.modeled else ""
            lines.append(
                f"  - {route.source}{aux} -> {route.destination}: "
                f"{route.amount:+.0f}%{bip}{curve}{tag}"
            )
    else:
        lines.append("Mod matrix: (no active routes)")

    lines.append("")
    porta = (
        f"  portamento={spec.global_.portamento_time:.2f}s" if spec.global_.portamento_time else ""
    )
    lines.append(
        f"Global: master_volume={spec.global_.master_volume:.2f}  "
        f"mono={'on' if spec.global_.mono else 'off'}  "
        f"poly={spec.global_.poly_count:.0f}{porta}"
    )

    if spec.arp:
        arp = spec.arp
        timing = "dotted" if arp.dotted else ("triplets" if arp.triplets else "straight")
        transpose = f"  transpose_shape={arp.transpose_shape}" if arp.transpose_shape else ""
        pattern = (
            f"  pattern={len(arp.pattern)} notes @ {arp.pattern_step_beats:.4f} beats/step"
            if arp.pattern
            else ""
        )
        lines.append(
            f"Arp: ON  shape={arp.shape}  rate={arp.rate:.2f}  gate={arp.gate:.0f}%  "
            f"{timing}  transpose_shift={arp.transpose_shift:+.0f}st{transpose}{pattern}"
        )

    return "\n".join(lines)
