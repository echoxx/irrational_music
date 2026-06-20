"""Cross-modulation: one irrational constant modulating another.

A modulator constant supplies a second digit stream that advances one digit
per carrier note. Each selected target maps the modulator digit (0-9,
normalized to t in [0, 1]) plus a global depth slider (0-1) onto a per-note
parameter of the carrier's note event (see synth.render_sequence for the
note-event format). Example: pi picks the pitches while e's digits set each
note's duration — an irrational groove.

Both playback paths share this module: the Generate path applies it while
building the note list, and the live path applies it at each note boundary
in the audio callback.
"""

from synth import WAVEFORM_CHOICES


def _scale_freqs(freqs, mult):
    if isinstance(freqs, (list, tuple)):
        return [f * mult for f in freqs]
    return freqs * mult


def apply_rhythm(note, t, depth):
    """Duration multiplier 1/3x..3x (at full depth), exponentially spread."""
    note["duration"] *= 3.0 ** (depth * (2.0 * t - 1.0))


def apply_transpose(note, t, depth):
    """Continuous pitch transpose up to +/- one octave at full depth."""
    note["freqs"] = _scale_freqs(note["freqs"], 2.0 ** (depth * (2.0 * t - 1.0)))


def apply_harmonic(note, t, depth):
    """Jump the note to one of its own harmonics (1..10) — digit picks which."""
    harmonic = 1 + int(round(t * 9 * depth))
    note["freqs"] = _scale_freqs(note["freqs"], harmonic)


def apply_dynamics(note, t, depth):
    """Per-note volume accents; depth=1 spans 0.15x..1x."""
    note["volume"] *= (1.0 - depth) + depth * (0.15 + 0.85 * t)


def apply_brightness(note, t, depth):
    """Add digit-controlled harmonics on top of the base waveform."""
    note["brightness"] = note.get("brightness", 0.0) + depth * t


def apply_waveform(note, t, depth):
    """Digit switches the waveform per note (depth acts as an on/off gate)."""
    if depth > 0:
        digit = int(round(t * 9))
        note["waveform"] = WAVEFORM_CHOICES[digit % len(WAVEFORM_CHOICES)]


def apply_vibrato(note, t, depth):
    """Digit sets vibrato depth (up to ~3% deviation) and speeds up the LFO."""
    note["vibrato_depth"] = depth * 0.03 * t
    note["vibrato_rate"] = 4.0 + 4.0 * t


def apply_pan(note, t, depth):
    """Digit places the note in the stereo field (-depth..+depth)."""
    note["pan"] = depth * (2.0 * t - 1.0)


MOD_TARGETS = {
    "rhythm": ("Rhythm (note duration)", apply_rhythm),
    "transpose": ("Pitch transpose (continuous, ±octave)", apply_transpose),
    "harmonic": ("Harmonic jump (digit → overtone)", apply_harmonic),
    "dynamics": ("Dynamics (per-note volume)", apply_dynamics),
    "brightness": ("Brightness (added harmonics)", apply_brightness),
    "waveform": ("Waveform switch (digit → wave)", apply_waveform),
    "vibrato": ("Vibrato (pitch LFO)", apply_vibrato),
    "pan": ("Stereo pan", apply_pan),
}

MOD_TARGET_CHOICES = [(label, key) for key, (label, _) in MOD_TARGETS.items()]


def apply_modulation(note, targets, mod_digit, depth):
    """Apply the selected modulation targets to one note event, in place."""
    t = (mod_digit % 10) / 9.0
    for key in targets:
        MOD_TARGETS[key][1](note, t, float(depth))


def build_note_sequence(digits, freq_table, base_params, chord_size=1, chord_step=2,
                        mod_digits=None, mod_targets=(), mod_depth=0.5):
    """
    Build the note-event list for synth.render_sequence from a carrier digit
    stream.

    digits: carrier digits (already pairs if the mode uses pairs)
    freq_table: digit → frequency table from build_frequency_table()
    base_params: shared per-note defaults (duration, volume, waveform, ...)
    chord_size/chord_step: 1 = single notes; >1 stacks extra voices every
        chord_step table degrees above the digit's note (wrapping)
    mod_digits/mod_targets/mod_depth: optional modulator stream + targets
    """
    table_len = len(freq_table)
    notes = []
    for i, d in enumerate(digits):
        idx = d % table_len
        if chord_size > 1:
            freqs = [freq_table[(idx + k * chord_step) % table_len] for k in range(chord_size)]
        else:
            freqs = freq_table[idx]
        note = dict(base_params)
        note["freqs"] = freqs
        if mod_digits is not None and mod_targets:
            apply_modulation(note, mod_targets, mod_digits[i % len(mod_digits)], mod_depth)
        notes.append(note)
    return notes
