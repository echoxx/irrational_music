"""Gradio UI for the irrational sonification project.

Run with: python app.py

Controls are grouped into tabs (Source / Tuning / Timbre / Modulation / FX /
Visuals) with save/load presets. Two playback paths:

- Generate: renders a stereo buffer (synth.py + modulation.py + effects.py),
  shows the spectrogram, and plays in the browser.
- Live: plays continuously on the host machine's speakers (live.py); slider
  changes are heard within ~50 ms, and the Visuals panel (oscilloscope,
  scrolling spectrogram, digit ticker) updates on a timer while running.

The pre-revision interface is preserved in app_classic.py (port 7861).
"""

import matplotlib
matplotlib.use("Agg")  # headless backend; Gradio renders the figure

from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from effects import apply_effects_offline
from irrational import (
    FREQUENCY_MODES,
    IRRATIONAL_CONSTANTS,
    add_spectrogram_colorbar,
    build_frequency_table,
    draw_spectrogram,
    get_irrational_digit_pairs,
    get_irrational_digits,
    mode_uses_pairs,
    style_dark_axis,
    style_dark_figure,
)
from live import LivePlayer
from modulation import MOD_TARGET_CHOICES, build_note_sequence
from presets import load_presets, save_preset
from synth import (FM_PRESET_CHOICES, WAVEFORM_CHOICES, render_sequence,
                   resolve_fm_ratio)

SAMPLE_RATE = 44100

live_player = LivePlayer()

CONSTANT_CHOICES = [(name, key) for key, (name, _, _) in IRRATIONAL_CONSTANTS.items()]
NONE_PLUS_CONSTANTS = [("None", "none")] + CONSTANT_CHOICES
MODE_CHOICES = [(label, key) for key, (label, _, _) in FREQUENCY_MODES.items()]


# =============================================================================
# SYNTHESIS (Generate path)
# =============================================================================

def build_spectrogram(audio, sample_rate, title, log_freq=False, fmax=3000):
    fig, ax = plt.subplots(figsize=(10, 4))
    style_dark_figure(fig)
    mesh = draw_spectrogram(ax, audio, sample_rate, title=title,
                            log_freq=log_freq, fmax=fmax)
    if mesh is not None:
        add_spectrogram_colorbar(fig, mesh, ax)
    fig.tight_layout()
    return fig


def get_digits_for_mode(constant, num_digits, mode):
    if mode_uses_pairs(mode):
        return get_irrational_digit_pairs(constant, num_digits)
    return get_irrational_digits(constant, num_digits)


def render_voice(constant, num_digits, mode, base_freq, subdivisions,
                 base_params, envelope, crossfade, adsr,
                 chord_size=1, chord_step=2,
                 mod_constant="none", mod_targets=(), mod_depth=0.5,
                 harm_constant="none", harm_slide=False, harm_offset=0,
                 harm_rolloff=0.5):
    """Render one voice (carrier or counterpoint) to a stereo buffer."""
    num_digits = int(num_digits)
    subdivisions = max(1, int(subdivisions))
    freq_table = build_frequency_table(mode, base_freq, subdivisions)
    digits = get_digits_for_mode(constant, num_digits, mode)

    mod_digits = None
    if mod_constant != "none" and mod_targets:
        # The modulator always advances one *single* digit per carrier note.
        mod_digits = get_irrational_digits(mod_constant, num_digits)

    harm_digits = None
    if harm_constant != "none":
        harm_digits = get_irrational_digits(harm_constant, num_digits)

    notes = build_note_sequence(
        digits, freq_table, base_params,
        chord_size=int(chord_size), chord_step=int(chord_step),
        mod_digits=mod_digits, mod_targets=tuple(mod_targets), mod_depth=mod_depth,
        harm_digits=harm_digits, harm_slide=bool(harm_slide),
        harm_offset=int(harm_offset), harm_rolloff=float(harm_rolloff),
    )
    return render_sequence(
        notes, sample_rate=SAMPLE_RATE,
        envelope=envelope, crossfade=crossfade, adsr=adsr, stereo=True,
    )


def mix_buffers(a, b):
    """Sum two stereo buffers, padding the shorter with silence."""
    n = max(len(a), len(b))
    out = np.zeros((n, 2), dtype=np.float32)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def synthesize(constant, num_digits, mode, base_freq, subdivisions, duration, crossfade, volume,
               waveform="sine", pulse_width=0.3, morph=0.0,
               brightness=0.0, fm_depth=0.0, fm_ratio=2.0, fm_preset="custom",
               harm_constant="none", harm_slide=False, harm_offset=0, harm_rolloff=0.5,
               envelope="crossfade", attack=0.005, decay=0.04, sustain=0.7, release=0.04,
               chord_size=1, chord_step=2, pan=0.0,
               mod_constant="none", mod_targets=(), mod_depth=0.5,
               cp_constant="none", cp_mode="harmonic_series", cp_base_freq=0,
               cp_waveform="sine", cp_volume=0.15, cp_pan=0.5, cp_duration=0,
               fx_chorus=0.0, fx_delay=0.0, fx_reverb=0.0,
               fx_room_size=0.5, fx_damping=0.5, fx_width=1.0, fx_predelay=0.0,
               log_freq=False):
    plt.close("all")  # release figures from previous calls
    duration = float(duration)
    crossfade = min(float(crossfade), duration / 2)
    adsr = (float(attack), float(decay), float(sustain), float(release))
    mod_targets = tuple(mod_targets or ())

    base_params = {
        "duration": duration,
        "volume": float(volume),
        "pan": float(pan),
        "waveform": waveform,
        "pulse_width": float(pulse_width),
        "morph": float(morph),
        "brightness": float(brightness),
        "fm_depth": float(fm_depth),
        "fm_ratio": resolve_fm_ratio(fm_preset, fm_ratio),
    }
    audio = render_voice(
        constant, num_digits, mode, base_freq, subdivisions,
        base_params, envelope, crossfade, adsr,
        chord_size=chord_size, chord_step=chord_step,
        mod_constant=mod_constant, mod_targets=mod_targets, mod_depth=float(mod_depth),
        harm_constant=harm_constant, harm_slide=harm_slide,
        harm_offset=harm_offset, harm_rolloff=harm_rolloff,
    )

    title_parts = [f"{IRRATIONAL_CONSTANTS[constant][0]} — {mode}, {int(num_digits)} "
                   f"{'pairs' if mode_uses_pairs(mode) else 'digits'}"]
    if mod_constant != "none" and mod_targets:
        title_parts.append(f"mod: {IRRATIONAL_CONSTANTS[mod_constant][0]} → {', '.join(mod_targets)}")

    if cp_constant != "none":
        cp_params = {
            "duration": float(cp_duration) if cp_duration else duration,
            "volume": float(cp_volume),
            "pan": float(cp_pan),
            "waveform": cp_waveform,
            "pulse_width": float(pulse_width),
            "morph": float(morph),
            "brightness": float(brightness),
            "fm_depth": float(fm_depth),
            "fm_ratio": resolve_fm_ratio(fm_preset, fm_ratio),
        }
        cp_audio = render_voice(
            cp_constant, num_digits, cp_mode,
            float(cp_base_freq) if cp_base_freq else base_freq, subdivisions,
            cp_params, envelope, crossfade, adsr,
        )
        audio = mix_buffers(audio, cp_audio)
        title_parts.append(f"vs {IRRATIONAL_CONSTANTS[cp_constant][0]}")

    audio = apply_effects_offline(
        audio, SAMPLE_RATE,
        chorus=float(fx_chorus), delay=float(fx_delay), reverb=float(fx_reverb),
        reverb_room=float(fx_room_size), reverb_damp=float(fx_damping),
        reverb_width=float(fx_width), reverb_predelay=float(fx_predelay),
    )

    title = "  |  ".join(title_parts)
    fig = build_spectrogram(audio, SAMPLE_RATE, title, log_freq=log_freq)

    return (SAMPLE_RATE, audio), fig


# =============================================================================
# LIVE VISUALS
# =============================================================================

def build_oscilloscope(buf):
    fig, ax = plt.subplots(figsize=(10, 1.8))
    style_dark_figure(fig)
    mono = buf[-2048:].mean(axis=1)
    t_ms = np.arange(len(mono)) / SAMPLE_RATE * 1000.0
    ax.plot(t_ms, mono, color="#7fd7ff", linewidth=0.8)
    style_dark_axis(ax)
    ax.set_xlim(0, t_ms[-1] if len(t_ms) else 1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("ms")
    ax.set_title("Oscilloscope (last 46 ms)")
    fig.tight_layout()
    return fig


def build_ticker(info):
    if not info or info.get("digit") is None:
        return "*(waiting for first note...)*"
    freqs = info.get("freqs") or []
    parts = [f"**Digit {info['digit']}** (position #{info['digit_index']}) → "
             + ", ".join(f"{f:g} Hz" for f in freqs)]
    if info.get("mod_digit") is not None:
        parts.append(f"modulator digit: **{info['mod_digit']}**")
    if info.get("cp_digit") is not None:
        parts.append(f"counterpoint digit: **{info['cp_digit']}**")
    return " &nbsp;•&nbsp; ".join(parts)


def visuals_tick(log_freq):
    if not live_player.is_running:
        return gr.update(), gr.update(), gr.update(), gr.update()
    plt.close("all")
    buf, info = live_player.get_visual_snapshot()
    osc = build_oscilloscope(buf)
    # Analyze only the recent tail: keeps the per-tick FFT/render cost (and
    # its GIL hold) small so the audio callback never starves.
    spec_fig, ax = plt.subplots(figsize=(10, 2.6))
    style_dark_figure(spec_fig)
    draw_spectrogram(ax, buf[-int(1.5 * SAMPLE_RATE):], SAMPLE_RATE,
                     title="Live (last 1.5 s)", log_freq=log_freq)
    spec_fig.tight_layout()
    # gr.update() (no-op) when not recording so we don't stomp the post-stop
    # "Recorded Xs" status message.
    rec_md = (f"🔴 **Recording… {live_player.recording_elapsed():.1f}s**"
              if live_player.is_recording else gr.update())
    return osc, spec_fig, build_ticker(info), rec_md


# =============================================================================
# LIVE CONTROL
# =============================================================================

# Every synthesize() input, in signature order. Used for Generate inputs and
# preset save/load. Live mode uses the same keys minus crossfade/log_freq.
PARAM_KEYS = [
    "constant", "num_digits", "mode", "base_freq", "subdivisions", "duration",
    "crossfade", "volume",
    "waveform", "pulse_width", "morph",
    "brightness", "fm_depth", "fm_ratio", "fm_preset",
    "harm_constant", "harm_slide", "harm_offset", "harm_rolloff",
    "envelope", "attack", "decay", "sustain", "release",
    "chord_size", "chord_step", "pan",
    "mod_constant", "mod_targets", "mod_depth",
    "cp_constant", "cp_mode", "cp_base_freq", "cp_waveform", "cp_volume",
    "cp_pan", "cp_duration",
    "fx_chorus", "fx_delay", "fx_reverb",
    "fx_room_size", "fx_damping", "fx_width", "fx_predelay",
    "log_freq",
]
# Live-only keys (not part of synthesize()'s signature; Generate plays the
# sequence once, so looping doesn't apply there).
LIVE_ONLY_KEYS = ["loop_mode"]
LIVE_KEYS = [k for k in PARAM_KEYS if k not in ("crossfade", "log_freq")] + LIVE_ONLY_KEYS
# Controls whose change requires re-fetching digit streams (mpmath work).
REFRESH_KEYS = {"constant", "num_digits", "mode", "mod_constant", "cp_constant", "cp_mode",
                "harm_constant"}


def start_live(*vals):
    params = dict(zip(LIVE_KEYS, vals))
    params["mod_targets"] = tuple(params.get("mod_targets") or ())
    live_player.set_params(**params)
    live_player.refresh_digits()
    try:
        live_player.start()
    except Exception as e:
        return (f"⚠️ **Could not start live audio:** {e}\n\n"
                "Check that an output device is available on the host machine, "
                "then try again."), gr.Timer(active=False)
    # second output activates the visuals timer
    return "**Live: running** (audio on the host machine's speakers)", gr.Timer(active=True)


def stop_live():
    live_player.stop()
    return "Live: stopped", gr.Timer(active=False)


def restart_live():
    if not live_player.is_running:
        return "Live: stopped — start Live first, then Restart rewinds it"
    live_player.restart_sequence()
    return "**Live: running** — restarted from the first digit"


def start_recording(*vals):
    """Begin a recorded performance (auto-starts the engine if needed)."""
    params = dict(zip(LIVE_KEYS, vals))
    params["mod_targets"] = tuple(params.get("mod_targets") or ())
    live_player.set_params(**params)
    live_player.refresh_digits()
    try:
        live_player.start_recording()
    except Exception as e:
        return (f"⚠️ **Could not start recording:** {e}\n\n"
                "Check that an output device is available on the host machine, "
                "then try again."), gr.Timer(active=False)
    return "🔴 **Recording…** (drag controls to perform)", gr.Timer(active=True)


def stop_recording():
    """Finalize the take; playback keeps running. Returns the captured track."""
    result = live_player.stop_recording()
    # Visuals timer stays active only while the engine is still playing.
    timer_update = gr.Timer(active=live_player.is_running)
    if result is None:
        return "Recording: nothing captured", gr.update(), timer_update

    audio, secs, truncated = result
    saved = datetime.now().strftime("performance_%Y%m%d_%H%M%S.wav")
    try:
        wavfile.write(saved, SAMPLE_RATE, audio)  # float32; .wav is gitignored
    except Exception:
        saved = None  # degrade gracefully — playback still works

    status = f"**Recorded {secs:.1f}s** — playable/downloadable →"
    if truncated:
        status += "  \n_(stopped at the 10-minute recording cap)_"
    if saved:
        status += f"  \nSaved `{saved}`"
    return status, (SAMPLE_RATE, audio), timer_update


def live_set(key, refresh_digits=False):
    """Build a Gradio change-handler that pushes a single param into the live player."""
    def handler(value):
        if not live_player.is_running:
            return
        if key == "mod_targets":
            value = tuple(value or ())
        live_player.set_param(key, value)
        if refresh_digits:
            live_player.refresh_digits()
    return handler


# =============================================================================
# UI
# =============================================================================

with gr.Blocks(title="Irrational Sonification") as demo:
    gr.Markdown(
        "# Irrational number sonification\n"
        "Map the digits of π, e, φ, √2, γ, Champernowne and more to frequencies — "
        "with waveforms, cross-modulation, counterpoint, and effects."
    )

    with gr.Row():
        with gr.Column(scale=1):
            # ---------------- presets
            with gr.Row():
                preset_dd = gr.Dropdown(choices=sorted(load_presets().keys()),
                                        label="Preset", scale=2, allow_custom_value=True)
                load_btn = gr.Button("Load", scale=1)
                save_btn = gr.Button("Save", scale=1)

            with gr.Tabs():
                with gr.Tab("Source"):
                    constant = gr.Dropdown(choices=CONSTANT_CHOICES, value="pi", label="Constant")
                    num_digits = gr.Slider(minimum=10, maximum=1000, value=100, step=10,
                                           label="Number of digits")
                    duration = gr.Slider(minimum=0.01, maximum=0.5, value=0.05, step=0.005,
                                         label="Note duration (s)")
                    volume = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="Volume")
                    pan = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.05,
                                    label="Pan (carrier voice)")
                    loop_mode = gr.Radio(
                        choices=[("Forward (jump back to start)", "forward"),
                                 ("Ping-pong (reverse at the ends — seamless)", "pingpong")],
                        value="forward", label="Loop mode (Live)",
                    )

                with gr.Tab("Tuning"):
                    mode = gr.Dropdown(choices=MODE_CHOICES, value="harmonic_series",
                                       label="Frequency mode")
                    base_freq = gr.Slider(minimum=110, maximum=880, value=220, step=1,
                                          label="Base / tone (Hz)")
                    subdivisions = gr.Slider(
                        minimum=2, maximum=10, value=2, step=1,
                        label="Microtonal subdivisions per semitone",
                        visible=False,
                    )

                with gr.Tab("Timbre"):
                    waveform = gr.Radio(choices=WAVEFORM_CHOICES + ["morph"], value="sine",
                                        label="Waveform ('morph' blends them continuously)")
                    pulse_width = gr.Slider(minimum=0.05, maximum=0.95, value=0.3, step=0.05,
                                            label="Pulse width", visible=False)
                    morph = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01,
                                      label="Morph (sine → tri → saw → square → pulse)",
                                      visible=False)
                    brightness = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                                           label="Brightness (added harmonics)")
                    fm_depth = gr.Slider(minimum=0.0, maximum=8.0, value=0.0, step=0.1,
                                         label="FM depth")
                    fm_ratio = gr.Slider(minimum=0.25, maximum=8.0, value=2.0, step=0.25,
                                         label="FM ratio (modulator/carrier)")
                    fm_preset = gr.Dropdown(choices=FM_PRESET_CHOICES, value="custom",
                                            label="FM ratio preset (irrational ratios → "
                                                  "inharmonic, bell-like timbres)")
                    envelope = gr.Radio(choices=[("Crossfade (smooth overlap)", "crossfade"),
                                                 ("ADSR (shaped notes)", "adsr")],
                                        value="crossfade", label="Envelope")
                    crossfade = gr.Slider(minimum=0.0, maximum=0.05, value=0.01, step=0.001,
                                          label="Crossfade (s) — Generate only")
                    with gr.Row(visible=False) as adsr_row:
                        attack = gr.Slider(minimum=0.001, maximum=0.2, value=0.005, step=0.001,
                                           label="Attack (s)")
                        decay = gr.Slider(minimum=0.0, maximum=0.3, value=0.04, step=0.005,
                                          label="Decay (s)")
                        sustain = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                                            label="Sustain")
                        release = gr.Slider(minimum=0.001, maximum=0.3, value=0.04, step=0.005,
                                            label="Release (s)")
                    with gr.Row():
                        chord_size = gr.Slider(minimum=1, maximum=4, value=1, step=1,
                                               label="Chord size (1 = single notes)")
                        chord_step = gr.Slider(minimum=1, maximum=4, value=2, step=1,
                                               label="Chord stacking step (scale degrees)")
                    gr.Markdown("---\n**Digit harmonics** — the digits of a constant set "
                                "the amplitudes of harmonics 1–16, so you hear its spectrum. "
                                "Overrides waveform, morph, and brightness (FM still applies).")
                    harm_constant = gr.Dropdown(choices=NONE_PLUS_CONSTANTS, value="none",
                                                label="Spectrum constant")
                    harm_slide = gr.Checkbox(value=False,
                                             label="Sliding window (spectrum evolves "
                                                   "note by note)")
                    with gr.Row():
                        harm_offset = gr.Slider(minimum=0, maximum=200, value=0, step=1,
                                                label="Window offset (digit index)")
                        harm_rolloff = gr.Slider(minimum=0.0, maximum=2.0, value=0.5, step=0.05,
                                                 label="Rolloff (1/k^r, tames harshness)")

                with gr.Tab("Modulation"):
                    gr.Markdown("One constant modulates another: the modulator's digits "
                                "steer per-note parameters of the carrier (e.g. π plays "
                                "pitches while e shapes the rhythm).")
                    mod_constant = gr.Dropdown(choices=NONE_PLUS_CONSTANTS, value="none",
                                               label="Modulator constant")
                    mod_targets = gr.CheckboxGroup(choices=MOD_TARGET_CHOICES,
                                                   label="Modulation targets")
                    mod_depth = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                          label="Modulation depth")
                    gr.Markdown("---\n**Counterpoint** — a second constant playing "
                                "simultaneously as its own voice.")
                    cp_constant = gr.Dropdown(choices=NONE_PLUS_CONSTANTS, value="none",
                                              label="Counterpoint constant")
                    cp_mode = gr.Dropdown(choices=MODE_CHOICES, value="harmonic_series",
                                          label="Counterpoint frequency mode")
                    with gr.Row():
                        cp_base_freq = gr.Slider(minimum=0, maximum=880, value=0, step=1,
                                                 label="Counterpoint base (Hz, 0 = follow carrier)")
                        cp_duration = gr.Slider(minimum=0.0, maximum=0.5, value=0.0, step=0.005,
                                                label="Counterpoint note duration (s, 0 = follow)")
                    cp_waveform = gr.Radio(choices=WAVEFORM_CHOICES, value="sine",
                                           label="Counterpoint waveform")
                    with gr.Row():
                        cp_volume = gr.Slider(minimum=0.0, maximum=1.0, value=0.15, step=0.01,
                                              label="Counterpoint volume")
                        cp_pan = gr.Slider(minimum=-1.0, maximum=1.0, value=0.5, step=0.05,
                                           label="Counterpoint pan")

                with gr.Tab("FX"):
                    fx_chorus = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                                          label="Chorus")
                    fx_delay = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                                         label="Delay (feedback echo)")
                    fx_reverb = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                                          label="Reverb")
                    with gr.Row():
                        fx_room_size = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                                 label="Reverb room size")
                        fx_damping = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                               label="Reverb damping (high-freq absorption)")
                    with gr.Row():
                        fx_width = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01,
                                             label="Reverb stereo width")
                        fx_predelay = gr.Slider(minimum=0.0, maximum=0.25, value=0.0, step=0.005,
                                                label="Reverb pre-delay (s)")

                with gr.Tab("Visuals"):
                    log_freq = gr.Checkbox(value=False,
                                           label="Logarithmic frequency axis (matches pitch perception)")
                    gr.Markdown("The oscilloscope, live spectrogram, and digit ticker on "
                                "the right update automatically while Live mode runs.")

            btn = gr.Button("Generate", variant="primary")
            gr.Markdown(
                "### Live mode\n"
                "Plays continuously on the host machine's speakers. "
                "Drag any control while running to hear changes immediately."
            )
            with gr.Row():
                start_btn = gr.Button("Start Live", variant="primary")
                restart_btn = gr.Button("⟲ Restart (digit 1)")
                stop_btn = gr.Button("Stop Live")
            live_status = gr.Markdown("Live: stopped")
            gr.Markdown(
                "### Record a performance\n"
                "Hit Record, then drag controls; the whole take is captured and "
                "saved as a track. Playback keeps running when you stop."
            )
            with gr.Row():
                record_btn = gr.Button("⏺ Record", variant="primary")
                stop_record_btn = gr.Button("Stop Record")
            record_status = gr.Markdown("Recording: idle")

        with gr.Column(scale=2):
            audio_out = gr.Audio(label="Audio", type="numpy")
            spec_out = gr.Plot(label="Spectrogram")
            recording_out = gr.Audio(label="Recorded performance", type="numpy")
            with gr.Accordion("Live visuals (populate while Live mode runs)", open=False):
                ticker_out = gr.Markdown("*(live digit ticker)*")
                osc_out = gr.Plot(label="Oscilloscope (live)")
                live_spec_out = gr.Plot(label="Live spectrogram")

    # ------------------------------------------------------------------ wiring
    components = {
        "constant": constant, "num_digits": num_digits, "mode": mode,
        "base_freq": base_freq, "subdivisions": subdivisions, "duration": duration,
        "crossfade": crossfade, "volume": volume,
        "waveform": waveform, "pulse_width": pulse_width, "morph": morph,
        "brightness": brightness,
        "fm_depth": fm_depth, "fm_ratio": fm_ratio, "fm_preset": fm_preset,
        "harm_constant": harm_constant, "harm_slide": harm_slide,
        "harm_offset": harm_offset, "harm_rolloff": harm_rolloff,
        "envelope": envelope, "attack": attack, "decay": decay, "sustain": sustain,
        "release": release,
        "chord_size": chord_size, "chord_step": chord_step, "pan": pan,
        "mod_constant": mod_constant, "mod_targets": mod_targets, "mod_depth": mod_depth,
        "cp_constant": cp_constant, "cp_mode": cp_mode, "cp_base_freq": cp_base_freq,
        "cp_waveform": cp_waveform, "cp_volume": cp_volume, "cp_pan": cp_pan,
        "cp_duration": cp_duration,
        "fx_chorus": fx_chorus, "fx_delay": fx_delay, "fx_reverb": fx_reverb,
        "fx_room_size": fx_room_size, "fx_damping": fx_damping,
        "fx_width": fx_width, "fx_predelay": fx_predelay,
        "log_freq": log_freq,
        "loop_mode": loop_mode,
    }
    synth_inputs = [components[k] for k in PARAM_KEYS]
    live_inputs = [components[k] for k in LIVE_KEYS]

    # conditional visibility
    mode.change(fn=lambda m: gr.update(visible=(m == "microtonal")),
                inputs=mode, outputs=subdivisions)
    waveform.change(
        fn=lambda w: (gr.update(visible=(w in ("pulse", "morph"))),
                      gr.update(visible=(w == "morph"))),
        inputs=waveform, outputs=[pulse_width, morph],
    )
    fm_preset.change(fn=lambda pr: gr.update(visible=(pr == "custom")),
                     inputs=fm_preset, outputs=fm_ratio)
    envelope.change(
        fn=lambda e: (gr.update(visible=(e == "adsr")), gr.update(visible=(e == "crossfade"))),
        inputs=envelope, outputs=[adsr_row, crossfade],
    )

    # Visuals timer: created inactive; Start/Stop Live toggles it. Hidden
    # progress so the plots don't grey out on every tick.
    timer = gr.Timer(0.8, active=False)

    btn.click(fn=synthesize, inputs=synth_inputs, outputs=[audio_out, spec_out])

    start_btn.click(fn=start_live, inputs=live_inputs, outputs=[live_status, timer])
    restart_btn.click(fn=restart_live, outputs=live_status)
    stop_btn.click(fn=stop_live, outputs=[live_status, timer])

    record_btn.click(fn=start_recording, inputs=live_inputs, outputs=[record_status, timer])
    stop_record_btn.click(fn=stop_recording,
                          outputs=[record_status, recording_out, timer])

    # Live-mode control wiring. `change` fires on every drag tick — fine
    # because set_param is just a locked dict update. num_digits uses `release`
    # to avoid re-running mpmath on every intermediate value.
    for key in LIVE_KEYS:
        comp = components[key]
        handler = live_set(key, refresh_digits=(key in REFRESH_KEYS))
        if key == "num_digits":
            comp.release(fn=handler, inputs=comp)
        else:
            comp.change(fn=handler, inputs=comp)

    # ---------------- presets
    def do_save(name, *vals):
        presets = save_preset(name, dict(zip(PARAM_KEYS, vals)))
        return gr.update(choices=sorted(presets.keys()), value=(name or "").strip())

    def do_load(name):
        cfg = load_presets().get(name)
        if not cfg:
            return [gr.update() for _ in PARAM_KEYS]
        return [gr.update(value=cfg[k]) if k in cfg else gr.update() for k in PARAM_KEYS]

    save_btn.click(fn=do_save, inputs=[preset_dd] + synth_inputs, outputs=preset_dd)
    load_btn.click(fn=do_load, inputs=preset_dd, outputs=synth_inputs)

    # ---------------- live visuals timer wiring
    timer.tick(fn=visuals_tick, inputs=log_freq,
               outputs=[osc_out, live_spec_out, ticker_out, record_status],
               show_progress="hidden")

    # Populate the page with one render of the defaults so it doesn't open
    # to empty placeholder boxes.
    demo.load(fn=synthesize, inputs=synth_inputs, outputs=[audio_out, spec_out])


if __name__ == "__main__":
    demo.launch(show_error=True)
