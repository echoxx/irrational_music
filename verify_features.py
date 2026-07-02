"""Regression checks for the timbre (morph / digit harmonics / irrational FM)
and Freeverb reverb features.

Run from the project dir with the Anaconda python:
    /mnt/e/anaconda3/python.exe verify_features.py

Each section asserts; the script prints PASS lines and exits non-zero on the
first failure.
"""

import inspect

import numpy as np

from effects import EffectChain
from irrational import get_irrational_digits
from modulation import build_note_sequence
from synth import (MORPH_ORDER, NUM_HARMONICS, FM_RATIO_PRESETS,
                   WAVEFORM_CHOICES, harmonic_amps, render_sequence,
                   render_wave, resolve_fm_ratio)

SR = 44100


def check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"PASS  {name}")


# ---------------------------------------------------------------- 1. signature
def check_signature():
    import app
    sig = list(inspect.signature(app.synthesize).parameters)
    check("synthesize() signature matches PARAM_KEYS order", sig == app.PARAM_KEYS)


# ------------------------------------------------------- 2. reverb block-split
def _run_chain(x, block_sizes, **amounts):
    chain = EffectChain(SR, x.shape[1])
    chain.set_amounts(**amounts)
    outs, i, s = [], 0, 0
    while i < len(x):
        n = min(block_sizes[s % len(block_sizes)], len(x) - i)
        outs.append(chain.process(x[i:i + n]))
        i += n
        s += 1
    return np.concatenate(outs)


def check_reverb():
    rng = np.random.default_rng(42)
    x = (rng.standard_normal((3 * SR, 2)) * 0.3).astype(np.float32)
    settings = dict(reverb=0.5, reverb_room=0.85, reverb_damp=0.3,
                    reverb_width=0.7, reverb_predelay=0.02)
    whole = _run_chain(x, [len(x)], **settings)
    check("reverb block-split (512) bit-identical",
          np.array_equal(whole, _run_chain(x, [512], **settings)))
    check("reverb block-split (irregular) bit-identical",
          np.array_equal(whole, _run_chain(x, [313, 1000, 2048], **settings)))
    extreme = dict(reverb=0.6, reverb_room=0.98, reverb_damp=0.9,
                   reverb_width=1.0, reverb_predelay=0.0)
    check("reverb block-split at extremes bit-identical",
          np.array_equal(_run_chain(x, [len(x)], **extreme),
                         _run_chain(x, [777], **extreme)))
    check("reverb output finite", np.isfinite(whole).all())

    chain = EffectChain(SR, 2)
    chain.set_amounts(reverb=1.0, reverb_room=0.5, reverb_damp=0.5)
    out = chain.process(x)
    dry = float(np.sqrt((x ** 2).mean()))
    wet = float(np.sqrt(((out - x) ** 2).mean()))
    print(f"      wet-path RMS @ wet=1 room=0.5: {wet:.4f} ({wet / dry:.2f}x dry)")
    check("reverb wet level in 0.2-0.6x dry range", 0.2 <= wet / dry <= 0.6)


# ------------------------------------------------------------ 3. timbre sweeps
def _render(notes_params, n_notes=30):
    freqs = [220.0 * 2 ** (d / 12) for d in range(10)]
    digits = get_irrational_digits("pi", n_notes)
    notes = [dict(notes_params, freqs=freqs[d % 10], duration=0.05) for d in digits]
    return render_sequence(notes, sample_rate=SR, stereo=False)


def check_stability():
    base = dict(volume=0.3)
    for wf in WAVEFORM_CHOICES:
        audio = _render(dict(base, waveform=wf))
        check(f"waveform '{wf}' finite & audible",
              np.isfinite(audio).all() and np.abs(audio).max() <= 1.5
              and np.sqrt((audio ** 2).mean()) > 1e-4)

    ph = 2 * np.pi * 220.0 * np.arange(SR) / SR
    for m, wf in [(0.0, "sine"), (0.25, "triangle"), (0.5, "sawtooth"),
                  (0.75, "square"), (1.0, "pulse")]:
        check(f"morph={m} lands exactly on {wf}",
              np.array_equal(render_wave(ph, "morph", morph=m),
                             render_wave(ph, wf)))
    for m in (0.1, 0.37, 0.62, 0.9):
        audio = _render(dict(base, waveform="morph", morph=m))
        check(f"morph={m} blend finite & audible",
              np.isfinite(audio).all() and np.abs(audio).max() <= 1.5
              and np.sqrt((audio ** 2).mean()) > 1e-4)

    for preset, ratio in FM_RATIO_PRESETS.items():
        check(f"fm preset '{preset}' resolves to {ratio:.5f}",
              resolve_fm_ratio(preset, 2.0) == ratio)
        audio = _render(dict(base, waveform="sine", fm_depth=4.0, fm_ratio=ratio))
        check(f"fm preset '{preset}' render finite & audible",
              np.isfinite(audio).all() and np.abs(audio).max() <= 1.5
              and np.sqrt((audio ** 2).mean()) > 1e-4)
    check("resolve_fm_ratio('custom') uses slider", resolve_fm_ratio("custom", 3.5) == 3.5)

    pi_digits = get_irrational_digits("pi", 100)
    for rolloff in (0.0, 0.5, 1.0):
        for slide in (False, True):
            notes = build_note_sequence(
                pi_digits[:30], [220.0 * 2 ** (d / 12) for d in range(10)],
                dict(base, duration=0.05, waveform="sine"),
                harm_digits=pi_digits, harm_slide=slide, harm_rolloff=rolloff)
            audio = render_sequence(notes, sample_rate=SR, stereo=False)
            check(f"harmonics rolloff={rolloff} slide={slide} finite & audible",
                  np.isfinite(audio).all() and np.abs(audio).max() <= 1.5
                  and np.sqrt((audio ** 2).mean()) > 1e-4)

    zero = render_wave(ph, harmonics=harmonic_amps([0] * NUM_HARMONICS))
    check("all-zero digit window falls back to fundamental (non-silent)",
          np.sqrt((zero ** 2).mean()) > 0.1)

    # morph modulation target produces per-note morph values
    from modulation import MOD_TARGETS
    note = {"waveform": "sine", "morph": 0.0}
    MOD_TARGETS["morph"][1](note, 7 / 9, 1.0)
    check("morph mod target sets waveform='morph' and morph=t",
          note["waveform"] == "morph" and abs(note["morph"] - 7 / 9) < 1e-9)


# ------------------------------------------------------------ 4. spectral check
def check_spectrum():
    digits = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]
    amps = harmonic_amps(digits, rolloff=0.0)
    f0, secs = 100.0, 2.0
    ph = 2 * np.pi * f0 * np.arange(int(secs * SR)) / SR
    sig = render_wave(ph, harmonics=amps)
    mag = np.abs(np.fft.rfft(sig))
    bins = [int(round(k * f0 * secs)) for k in range(1, NUM_HARMONICS + 1)]
    measured = np.array([mag[b] for b in bins])
    expected = amps / amps.max()
    err = np.abs(measured / measured.max() - expected).max()
    check(f"harmonic magnitudes match digits (max err {err:.4f})", err < 0.05)

    zero_digits = list(digits)
    zero_digits[2] = 0  # kill harmonic 3
    sig0 = render_wave(ph, harmonics=harmonic_amps(zero_digits, rolloff=0.0))
    mag0 = np.abs(np.fft.rfft(sig0))
    supp = 20 * np.log10(mag0[bins[2]] / mag0.max())
    check(f"zero-digit partial suppressed ({supp:.1f} dB)", supp < -40)


# --------------------------------------------------------------- 5. live smoke
def check_live():
    from live import LivePlayer, _Voice

    vp = {
        "duration": 0.05, "volume": 0.3, "pan": 0.0,
        "waveform": "morph", "pulse_width": 0.3, "morph": 0.4,
        "brightness": 0.0, "fm_depth": 0.0, "fm_ratio": 2.0,
        "harm_slide": True, "harm_offset": 0, "harm_rolloff": 0.5,
        "envelope": "crossfade", "adsr": (0.005, 0.04, 0.7, 0.04),
        "chord_size": 1, "chord_step": 2,
        "mod_targets": (), "mod_depth": 0.0, "loop_mode": "forward",
    }
    digits = get_irrational_digits("pi", 50)
    table = [220.0 * 2 ** (d / 12) for d in range(10)]

    voice = _Voice(SR)
    a = voice.render(2048, digits, [], digits, vp, table)
    b = voice.render(2048, digits, [], digits, vp, table)
    check("live voice (morph+harmonics) blocks finite",
          np.isfinite(a).all() and np.isfinite(b).all())

    # block-split continuity: fresh voice rendering 4096 == 2048+2048 voice
    voice2 = _Voice(SR)
    whole = voice2.render(4096, digits, [], digits, vp, table)
    split = np.concatenate([a, b])
    check("live voice block-split matches whole render",
          np.allclose(whole, split, atol=1e-4))

    # pure-sine seam: no phase discontinuity across block boundary
    vp_sine = dict(vp, waveform="sine")
    voice3 = _Voice(SR)
    s = np.concatenate([voice3.render(2048, digits, [], [], vp_sine, table),
                        voice3.render(2048, digits, [], [], vp_sine, table)])
    seam_step = abs(float(s[2048, 0]) - float(s[2047, 0]))
    check(f"sine seam step {seam_step:.5f} below slope bound", seam_step < 0.05)

    player = LivePlayer()
    player.set_params(harm_constant="e")
    player.refresh_digits()
    check("LivePlayer harm digit stream fetched",
          len(player.harm_digits) >= int(player.params["num_digits"])
          and player.harm_digits_key == ("e", int(player.params["num_digits"])))

    # restart_sequence: drive the callback directly (no audio stream needed),
    # advance a few notes, rewind, and confirm the digit walk restarted.
    player2 = LivePlayer()
    player2.refresh_digits()
    out = np.zeros((2048, 2), dtype=np.float32)
    for _ in range(6):
        player2._callback(out, 2048, None, None)
    check("live callback runs clean", player2.last_callback_error is None)
    before = player2.voice.digit_index
    player2.restart_sequence()
    player2._callback(out, 2048, None, None)
    after = player2.voice.digit_index
    check(f"restart rewinds the digit walk ({before} -> {after})",
          before >= 4 and after <= 2)


if __name__ == "__main__":
    check_reverb()
    check_stability()
    check_spectrum()
    check_live()
    check_signature()  # imports app (builds the Blocks UI) — keep last
    print("\nAll checks passed.")
