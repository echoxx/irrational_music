"""Shared synthesis engine for the irrational sonification project.

Single source of truth for waveform rendering, envelopes, and offline
note-sequence rendering. Both playback paths use it:

- the Generate path (irrational.generate_audio / app.py) renders whole
  buffers via render_sequence();
- the live path (live.LivePlayer) calls render_wave() with its own
  phase-continuous phase arrays so mid-stream parameter changes don't click.

All waveforms are plain 2*pi-periodic functions of phase, so the same
render_wave() works for both fresh per-note phases and a running phase
accumulator. Naive (non-band-limited) sawtooth/square will alias at high
pitches — accepted for this experimental tool; the additive 'brightness'
control offers a cleaner way to enrich a sine.
"""

import numpy as np
from scipy import signal as sps

WAVEFORM_CHOICES = ["sine", "sawtooth", "square", "triangle", "pulse"]

# Default ADSR used when envelope mode is 'adsr' and no override is given.
DEFAULT_ADSR = (0.005, 0.04, 0.7, 0.04)  # attack, decay, sustain level, release


def render_wave(phase, waveform="sine", pulse_width=0.3, brightness=0.0,
                fm_depth=0.0, fm_ratio=2.0):
    """
    Render samples from a phase array (radians).

    Parameters:
    phase (np.array): Instantaneous phase in radians (any offset is fine)
    waveform (str): One of WAVEFORM_CHOICES
    pulse_width (float): Duty cycle for 'pulse' (0..1)
    brightness (float): 0 = pure waveform; >0 adds harmonics k=2..8 at
        amplitude brightness/k (then renormalized), enriching the spectrum
    fm_depth (float): Phase-modulation index in radians (0 = off)
    fm_ratio (float): Modulator/carrier frequency ratio for FM

    Returns:
    np.array (float32): Samples in [-1, 1]
    """
    phase = np.asarray(phase, dtype=np.float64)
    if fm_depth > 0.0:
        phase = phase + fm_depth * np.sin(fm_ratio * phase)

    if waveform == "sine":
        out = np.sin(phase)
    elif waveform == "sawtooth":
        out = sps.sawtooth(phase)
    elif waveform == "square":
        out = sps.square(phase)
    elif waveform == "triangle":
        out = sps.sawtooth(phase, width=0.5)
    elif waveform == "pulse":
        out = sps.square(phase, duty=np.clip(pulse_width, 0.05, 0.95))
    else:
        raise ValueError(f"Unknown waveform '{waveform}'. Choose from: {WAVEFORM_CHOICES}")

    if brightness > 0.0:
        total = 1.0
        for k in range(2, 9):
            amp = brightness / k
            out = out + amp * np.sin(k * phase)
            total += amp
        out = out / total

    return out.astype(np.float32)


def adsr_envelope(num_samples, sample_rate, attack=0.005, decay=0.04,
                  sustain=0.7, release=0.04):
    """
    Per-note ADSR gain curve. Attack/decay/release are in seconds; sustain is
    a level 0-1. If the segments don't fit in num_samples they are squeezed
    proportionally so the note always ends at zero gain.
    """
    n_a = int(attack * sample_rate)
    n_d = int(decay * sample_rate)
    n_r = max(1, int(release * sample_rate))
    used = n_a + n_d + n_r
    if used > num_samples:
        scale = num_samples / used
        n_a = int(n_a * scale)
        n_d = int(n_d * scale)
        n_r = max(1, num_samples - n_a - n_d)
    n_s = max(0, num_samples - n_a - n_d - n_r)

    env = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, sustain, n_d, endpoint=False),
        np.full(n_s, sustain),
        np.linspace(sustain, 0.0, n_r),
    ])
    if len(env) < num_samples:
        env = np.pad(env, (0, num_samples - len(env)))
    return env[:num_samples].astype(np.float32)


def pan_gains(pan):
    """Equal-power stereo gains for pan in [-1 (left), +1 (right)]."""
    angle = (np.clip(pan, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return np.cos(angle), np.sin(angle)


def render_note(freqs, num_samples, sample_rate, waveform="sine",
                pulse_width=0.3, brightness=0.0, fm_depth=0.0, fm_ratio=2.0,
                vibrato_depth=0.0, vibrato_rate=5.0):
    """
    Render one (possibly chordal) note as mono float32.

    freqs: a single frequency or a list of frequencies summed as a chord
    vibrato_depth: fractional frequency deviation (e.g. 0.01 = +/-1%)
    """
    if np.isscalar(freqs):
        freqs = [freqs]
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    out = np.zeros(num_samples, dtype=np.float32)
    for freq in freqs:
        if vibrato_depth > 0.0:
            inst = freq * (1.0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t))
            phase = 2.0 * np.pi * np.cumsum(inst) / sample_rate
        else:
            phase = 2.0 * np.pi * freq * t
        out += render_wave(phase, waveform, pulse_width, brightness, fm_depth, fm_ratio)
    return out / max(1, len(freqs))


def render_sequence(notes, sample_rate=44100, envelope="crossfade",
                    crossfade=0.01, adsr=None, stereo=False):
    """
    Render a sequence of note events into one buffer.

    Each note is a dict with keys (all optional except freqs/duration):
      freqs (float | list) — frequency or chord frequencies
      duration (float)     — seconds
      volume (float)       — per-note amplitude (default 0.3)
      pan (float)          — -1..+1, only used when stereo=True (default 0)
      waveform, pulse_width, brightness, fm_depth, fm_ratio,
      vibrato_depth, vibrato_rate — per-note timbre (see render_note)

    envelope: 'crossfade' overlaps notes with cos^2 fades (the classic
    behavior); 'adsr' shapes each note with adsr_envelope and places notes
    back-to-back.

    Returns float32 array of shape (N,) or (N, 2) when stereo=True.
    """
    if not notes:
        shape = (0, 2) if stereo else (0,)
        return np.zeros(shape, dtype=np.float32)

    note_samples = [max(1, int(sample_rate * n["duration"])) for n in notes]

    if envelope == "adsr":
        xf = 0
    else:
        xf = int(sample_rate * crossfade)
        xf = min(xf, min(note_samples) // 2)

    starts = []
    pos = 0
    for ns in note_samples:
        starts.append(pos)
        pos += ns - xf
    total = starts[-1] + note_samples[-1]

    shape = (total, 2) if stereo else (total,)
    audio = np.zeros(shape, dtype=np.float32)

    fade_out = np.cos(np.linspace(0, np.pi / 2, xf)) ** 2 if xf else None
    fade_in = np.sin(np.linspace(0, np.pi / 2, xf)) ** 2 if xf else None

    for i, (note, start, ns) in enumerate(zip(notes, starts, note_samples)):
        tone = render_note(
            note["freqs"], ns, sample_rate,
            waveform=note.get("waveform", "sine"),
            pulse_width=note.get("pulse_width", 0.3),
            brightness=note.get("brightness", 0.0),
            fm_depth=note.get("fm_depth", 0.0),
            fm_ratio=note.get("fm_ratio", 2.0),
            vibrato_depth=note.get("vibrato_depth", 0.0),
            vibrato_rate=note.get("vibrato_rate", 5.0),
        )
        tone = tone * float(note.get("volume", 0.3))

        if envelope == "adsr":
            a, d, s, r = adsr or DEFAULT_ADSR
            tone *= adsr_envelope(ns, sample_rate, a, d, s, r)
        elif xf and i > 0:
            # fade this note in; the matching fade-out of the previous
            # note's tail is applied to the buffer just before adding.
            tone[:xf] *= fade_in

        if stereo:
            gl, gr = pan_gains(float(note.get("pan", 0.0)))
            if envelope != "adsr" and xf and i > 0:
                audio[start:start + xf, :] *= fade_out[:, None]
            audio[start:start + ns, 0] += tone * gl
            audio[start:start + ns, 1] += tone * gr
        else:
            if envelope != "adsr" and xf and i > 0:
                audio[start:start + xf] *= fade_out
            audio[start:start + ns] += tone

    return audio
