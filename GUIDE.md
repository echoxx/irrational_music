# Irrational Sonification — User Guide

This project turns the digits of mathematical constants into music. This guide explains what every control does and the ideas behind them. (For developer/architecture notes, see `CLAUDE.md`.)

## The core idea

An irrational constant like π has an infinite, never-repeating stream of decimal digits: 3, 1, 4, 1, 5, 9, 2, 6, …. The app walks that stream one digit at a time and plays a note for each: the **digit picks the pitch** from a 10-entry frequency table (digit 0 → table entry 0, digit 7 → table entry 7, and so on). Everything else in the UI shapes *how* that walk sounds — which frequencies the table holds (Tuning), what the notes are made of (Timbre), how a second constant can steer the first (Modulation), and the space it plays in (FX).

Because the digits never repeat, the melody never loops — but it isn't random either. Each constant has its own statistical fingerprint, and some (like Champernowne's counting constant) have an audibly obvious structure.

## Three ways to play

- **Generate** — renders the whole sequence to a fixed stereo clip, shows its spectrogram, and plays it in the browser. Use this for repeatable results you can download or share.
- **Start Live / Stop Live** — a real-time synth engine that plays continuously on the machine running the app (not in the browser). Drag any control while it runs and you hear the change within ~50 ms, without the sequence restarting. **⟲ Restart (digit 1)** rewinds the sequence to the constant's first digit while playing (the note currently sounding finishes, so the rewind is click-free; the counterpoint voice rewinds too). While live, the Visuals panel shows an oscilloscope, a scrolling spectrogram, and a ticker of the digits currently sounding.
- **Record / Stop Record** — captures a live *performance*: hit Record (it auto-starts the live engine if needed), drag controls to perform, hit Stop Record. The take appears in a browser player and is auto-saved as `performance_YYYYMMDD_HHMMSS.wav` in the project folder. There's a 10-minute cap per take; stopping the recording leaves playback running.

**Presets** (top of the left column) save and restore every control as a named entry in `presets.json`. Presets saved before a feature existed still load — new controls just keep their current values.

---

## Source tab — what plays

- **Constant** — which number's digits drive the melody. The 20 available:

  | Constant | Value | Notes |
  |---|---|---|
  | Pi (π) | 3.14159… | Circle circumference / diameter; the classic choice |
  | e | 2.71828… | Base of natural logarithms |
  | √2 | 1.41421… | First number ever proven irrational |
  | 12th root of 2 | 1.05946… | The semitone ratio of equal temperament — a tuning constant playing tunes |
  | φ (golden ratio) | 1.61803… | (1+√5)/2, the "most irrational" number |
  | √3, √5, √7 | 1.732…, 2.236…, 2.645… | Square roots of primes |
  | ∛2 | 1.25992… | Cube root of 2 |
  | ln 2, ln 10 | 0.693…, 2.302… | Natural logarithms |
  | γ (Euler–Mascheroni) | 0.57721… | Gap between the harmonic series and ln n; irrationality still unproven |
  | Catalan's constant | 0.91596… | From combinatorics; irrationality unproven |
  | Apéry's constant ζ(3) | 1.20205… | Sum of 1/n³; proven irrational in 1978 |
  | Khinchin's constant | 2.68545… | Geometric mean of continued-fraction terms of almost every real number |
  | e^π (Gelfond's) | 23.1406… | Proven transcendental |
  | π^π | 36.4621… | Not even proven irrational — but its digits play fine |
  | Silver ratio 1+√2 | 2.41421… | φ's lesser-known sibling |
  | Golden angle | 137.507… | 360°/φ², the sunflower-seed angle |
  | Champernowne | 0.1234567891011… | Digits literally count upward — sounds like ascending ramps, a great "control group" |

  The digit stream includes the integer part (π starts 3, 1, 4, …).

- **Number of digits** — sequence length (10–1000). In Live mode this is also the loop length.
- **Note duration** — seconds per note (0.01–0.5). Short = burbling texture, long = melody.
- **Volume** — master per-note amplitude.
- **Pan** — stereo position of the main voice (−1 left … +1 right), equal-power so loudness stays constant.
- **Loop mode (Live only)** — what happens when the live sequence reaches the last digit: **Forward** jumps back to digit 1 (an audible seam), **Ping-pong** reverses direction at the ends so the loop point is seamless.

## Tuning tab — which frequencies the digits pick

- **Frequency mode** — how the 10 digit slots (or 100, see "digit pairs") map to pitches:
  - **Harmonic series** — digits pick the 1st–10th harmonics of the base (base, 2×, 3×, … 10×). Everything is consonant with the fundamental; sounds organ-/bugle-like.
  - **Equal temperament** — 10 equal steps of one octave (not the usual 12 — a slightly alien even division).
  - **Continuous (digit pairs)** — reads digits **two at a time** (00–99) into a 100-entry table sweeping smoothly from base/2 to base×4. Effectively 100 microtonal pitches; melodies become contours rather than scales.
  - **Microtonal** — equal temperament with each semitone split into N **subdivisions** (the slider appears for this mode): quarter-tones and finer.
  - **Major / Minor / Major pentatonic / Chromatic scale** — digits pick degrees of a familiar 12-TET scale; degrees past the octave wrap upward. The most conventionally "musical" modes.
  - **Just intonation** — pure small-integer ratios (9/8, 5/4, 3/2, …). Beat-free, maximally consonant intervals.
  - **Bohlen–Pierce** — 13 equal divisions of the 3:1 "tritave" instead of the 2:1 octave. Alien but internally consistent.
  - **Pythagorean** — stacked pure 3:2 fifths folded into one octave. Pure fifths, characteristically wide thirds.
  - **Golden-ratio tuning** — powers of φ folded into one octave. Since φ is irrational, no pitch ever coincides with any equal temperament — a shimmering non-repeating scale (in-theme: an irrational number tuning the scale that plays another irrational number).
  - **Prime harmonics** — only prime-numbered partials (2, 3, 5, 7, 11, …). Hollow, bell-like.
  - **Inharmonic / bell** — stretched partials (base × n^1.3), like a struck bell or metal bar.
- **Base / tone (Hz)** — the root frequency everything is built on (110–880 Hz).

## Timbre tab — what each note is made of

- **Waveform** — the oscillator shape: **sine** (pure tone, no overtones), **sawtooth** (bright, brassy — all harmonics), **square** (hollow, clarinet-like — odd harmonics), **triangle** (soft, muted odd harmonics), **pulse** (like square but asymmetric; the **Pulse width** slider sets the duty cycle — thinner = more nasal), and **morph** *(new)*:
- **Morph** *(new)* — appears when waveform is "morph". One continuous slider that crossfades through sine → triangle → sawtooth → square → pulse. 0.25 is exactly a triangle, 0.37 is triangle-leaning-sawtooth, etc. Great to perform with in Live mode, and it's also a Modulation target (a constant's digits can sweep it per note).
- **Brightness** — additively mixes in harmonics 2–8 of the current note at decreasing strength. A cleaner way to enrich a sine than switching to sawtooth (no aliasing).
- **FM depth** — frequency modulation: a second (inaudible) oscillator wobbles the phase of the main one, creating sidebands. 0 = off; higher = increasingly metallic/clangorous.
- **FM ratio** — the modulator's frequency as a multiple of the note's. Integer ratios (2, 3…) give harmonic, musical spectra.
- **FM ratio preset** *(new)* — replaces the ratio slider with an exact irrational ratio: **π, φ, √2, or e**. Irrational ratios make the FM sidebands land *between* the harmonics — inharmonic, bell/gong-like timbres. In-theme twist: the constants shape the overtones themselves, not just the melody. "Custom" returns control to the slider.
- **Digit harmonics** *(new)* — the most literal sonification in the app: pick a **Spectrum constant** and its digits become the loudness of the note's harmonics. Digit window 3,1,4,1,5,9,… means harmonic 1 at strength 3/9, harmonic 2 at 1/9, harmonic 3 at 4/9, … through harmonic 16 — you are hearing the *spectrum* of π, not just its melody. Options:
  - **Sliding window** — the 16-digit window advances one digit per note, so the tone color itself evolves along the constant as the melody plays.
  - **Window offset** — start the window deeper into the digit stream.
  - **Rolloff (1/kʳ)** — attenuates higher harmonics by 1/k^r to tame harshness (0 = raw digits, 2 = very mellow).
  - When active, this **overrides** waveform/morph/brightness (the spectrum is fully specified by the digits); FM still applies on top. An all-zero window plays a pure fundamental rather than silence.
- **Envelope** — the loudness shape of each note: **Crossfade** overlaps each note's tail with the next note's start (smooth, legato; the **Crossfade** slider sets the overlap, Generate only), **ADSR** shapes each note with **Attack / Decay / Sustain / Release** sliders (percussive to pad-like, notes placed back-to-back).
- **Chord size / stacking step** — 1 = single notes; 2–4 stacks extra voices on top of each digit's note, each one **step** scale-degrees higher in the current tuning table (wrapping). Turns the digit walk into parallel harmony.

## Modulation tab — one constant steering another

A second constant's digit stream advances in lockstep with the melody (one modulator digit per note) and pushes the selected per-note parameters around. Example: π picks the pitches while e shapes the rhythm — an irrational groove that never repeats.

- **Modulator constant** — the steering digit stream ("None" = off).
- **Modulation targets** — what each modulator digit pushes (any combination):
  - **Rhythm** — note duration ×⅓…×3 (digit 0 = shortest, 9 = longest)
  - **Pitch transpose** — continuous ± one octave
  - **Harmonic jump** — jumps the note to one of its own overtones (digit picks which)
  - **Dynamics** — per-note volume accents
  - **Brightness** — digit adds harmonics on top of the base waveform
  - **Waveform switch** — digit picks one of the five discrete waveforms per note
  - **Wavetable morph** *(new)* — digit sweeps the continuous morph position per note (timbre ripples along the modulator's digits)
  - **Vibrato** — digit sets vibrato depth and speeds the LFO
  - **Stereo pan** — digit places the note in the stereo field
- **Modulation depth** — global 0–1 intensity for all targets.
- **Counterpoint** — a *second complete voice* playing simultaneously: its own constant, its own tuning mode and base frequency (0 = follow the carrier), waveform, volume, pan, and note duration (0 = follow). Set duration different from the carrier's for polyrhythms (e.g. π at 0.05 s against e at 0.075 s = 3:2).

## FX tab — the space it plays in

All effects run identically in Generate and Live modes.

- **Chorus** — a slightly delayed, slowly wobbling copy mixed in; thickens and widens (the two channels wobble in opposite phase for stereo width).
- **Delay** — feedback echo (~0.28 s, repeats fading at 45%).
- **Reverb** *(upgraded)* — now a Freeverb-style room simulation (8 damped comb filters + 4 allpass diffusers per channel) instead of the previous fixed small room. The main slider is still the wet amount; four new controls shape the room:
  - **Room size** — how long the tail rings: 0 ≈ a small booth, 1 ≈ a 10-second cathedral wash. (Generate automatically pads the clip so long tails ring out.)
  - **Damping** — high-frequency absorption, like soft vs. hard walls: low = bright and metallic, high = warm and muffled.
  - **Stereo width** — 1 = fully decorrelated spacious wet signal, 0 = mono reverb.
  - **Pre-delay (s)** — a gap between the dry note and the start of its reverb; even 20–40 ms keeps fast digit-runs articulate inside a big room. (Changing it mid-Live jumps the wet signal slightly — harmless.)

## Visuals tab & the right-hand panel

- **Logarithmic frequency axis** — plots spectrograms on a log axis, matching pitch perception (equal musical intervals get equal visual spacing).
- After **Generate**: the clip's full spectrogram.
- While **Live** runs: a digit ticker (current digit, its position, the frequencies sounding, plus modulator/counterpoint digits), an oscilloscope of the last 46 ms, and a scrolling spectrogram of the last 1.5 s.

---

## Other ways to run

- **CLI** (`./run.sh` / `python irrational.py`) — plays a hardcoded demo sequence of constants with spectrograms; sine only, no UI features.
- **Classic UI** (`./run_ui_classic.sh`, port 7861) — the older single-page interface, kept for A/B comparison; can run alongside the new UI.
- **Docker sandbox** (`./run-sandbox.sh`) — isolated environment with only this folder mounted; `--audio` routes sound to the host speakers, `--ui` runs the Gradio app. See `CLAUDE.md` for details.
- **`verify_features.py`** (run with the project's Anaconda python) — the regression check suite: verifies the reverb's block-split correctness, every timbre mode's stability, the digit-harmonics spectrum against the digits by FFT, live-engine continuity, and the UI parameter wiring.

## Quick recipes

- **Hear the spectrum of π**: Timbre → Digit harmonics → Spectrum constant = Pi, Sliding window on, Rolloff 0.5. Long-ish notes (0.2 s), ADSR envelope.
- **Irrational bells**: FM depth ≈ 4, FM preset = φ, tuning = Inharmonic/bell, big reverb (Room 0.8, Damping 0.3, Pre-delay 0.03).
- **π vs e polyrhythm**: Counterpoint constant = e, counterpoint duration 0.075 vs carrier 0.05, pan them apart.
- **Evolving texture to perform**: Waveform = morph, Start Live, ride the Morph and Room size sliders; hit Record to keep the take.
- **A tune that counts**: Constant = Champernowne, Major pentatonic — you'll hear the digits counting up.
