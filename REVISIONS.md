# Revision Log

## 2026-06-11 — Post-revision fixes (same day)

- **Glitches during live playback (perceived at sequence restart)**: the
  rendered waveform was verified smooth at the wrap; the audible glitches
  were buffer underruns — the visuals tick (FFT + matplotlib + PNG encode)
  holds the GIL up to ~20 ms, occasionally starving the audio callback.
  Fixes: `latency="high"` on the output stream (more device buffering;
  param response ~0.1 s), the live spectrogram now analyzes 1.5 s instead
  of 3 s, and the visuals timer ticks at 0.8 s instead of 0.5 s.
- **New "Loop mode" control (Source tab, live only)**: `forward` (wrap to
  the start, as before) or `pingpong` — the digit sequence reverses
  direction at the ends so the loop seam has no jump at all. The modulator
  stream follows the same loop shape.

- **UI looked stuck on load**: the live-visuals `gr.Timer` ticked from page
  open and re-greyed its output plots every 0.5 s. The timer is now created
  inactive, toggled by Start/Stop Live, and ticks with hidden progress.
- **Page opened to empty boxes**: `demo.load()` now runs one Generate with
  the default settings, so the Audio player and Spectrogram are populated as
  soon as the page opens; the live-only visuals moved into a collapsed
  accordion.
- **Start Live failed with `PortAudioError: Error querying device -1`**:
  PortAudio caches its device scan at first init, and a long-running server
  process can end up with a stale/empty list (e.g. the Windows default
  output changed). `LivePlayer.start()` now validates the default output and
  re-initializes PortAudio (`sd._terminate()/_initialize()`) to rescan,
  falling back to the first stereo-capable output. Live-start errors are now
  also reported in the UI status line instead of failing silently
  (`show_error=True` + try/except in `start_live`).

## 2026-06-11 — Big revision: timbre, modulation, counterpoint, live visuals, tabbed UI

The pre-revision interface is preserved as `app_classic.py` (`./run_ui_classic.sh`, port 7861)
so old and new can be compared side by side.

### New modules
- **`synth.py`** — shared synthesis engine for both playback paths:
  `render_wave()` (sine / sawtooth / square / triangle / pulse, additive
  "brightness" harmonics, FM), `adsr_envelope()`, `pan_gains()`, and
  `render_sequence()` — an offline note-event renderer with per-note timbre,
  chords, vibrato, stereo pan, and crossfade-or-ADSR envelopes.
- **`modulation.py`** — cross-modulation: a second constant's digit stream
  steers per-note parameters of the carrier. Targets: rhythm (duration),
  transpose (±octave), harmonic jump, dynamics, brightness, waveform switch,
  vibrato, stereo pan; global depth control. (e.g. π plays pitches while e
  shapes the rhythm.)
- **`effects.py`** — streamable chorus, feedback delay, and Schroeder reverb
  (`EffectChain`). Block-by-block processing is bit-identical to whole-buffer
  processing, so Generate and Live sound the same.
- **`presets.py`** — save/load named UI presets to `presets.json` (gitignored).
- **`app_classic.py` + `run_ui_classic.sh`** — snapshot of the previous UI.

### irrational.py
- Constants registry grew 7 → 20: added √5, √7, ∛2, ln(10), γ (Euler–
  Mascheroni), Catalan, Apéry ζ(3), Khinchin, e^π (Gelfond), π^π, silver
  ratio, golden angle, Champernowne. All getters share `_mp_digits()`.
  (Note: Khinchin replaces the originally-proposed Feigenbaum δ — mpmath
  cannot compute Feigenbaum to arbitrary precision.)
- Frequency modes grew 4 → 14: added major / minor / major-pentatonic /
  chromatic scales, just intonation (5-limit), Bohlen–Pierce (13 steps of
  3:1), Pythagorean (stacked fifths), golden-ratio tuning, prime harmonics,
  and inharmonic (stretched, bell-like) partials.
- New `FREQUENCY_MODES` registry + `build_frequency_table()` /
  `mode_uses_pairs()` — single source of truth used by both `app.py` and
  `live.py` (previously duplicated if/elif chains).
- `generate_audio()` now delegates to `synth.render_sequence()` and accepts
  waveform / brightness / FM / envelope kwargs (back-compatible defaults).
- Spectrograms restyled: shared `draw_spectrogram()` helper — dark theme,
  magma colormap, peak-normalized dB with −80 dB floor, optional log-frequency
  axis, subtle grid. Used by the CLI plots and the UI alike.

### live.py (rewritten)
- Stereo output; per-voice `_Voice` sequencer with persistent oscillator
  phases (click-free parameter changes preserved).
- Full feature parity with Generate: waveforms/brightness/FM, ADSR or 2 ms
  micro-fade envelopes, chords, cross-modulation, an independent counterpoint
  voice, and the streaming effects chain.
- Rolling ~3 s buffer + current-digit info exposed via
  `get_visual_snapshot()` for the UI visuals (oscilloscope, live spectrogram,
  digit ticker).
- Measured ~1.9 ms to render a 46.4 ms block with every feature enabled.

### app.py (rebuilt)
- Tabbed layout: Source / Tuning / Timbre / Modulation (incl. counterpoint) /
  FX / Visuals, with a save/load preset row.
- Generate path renders stereo, supports modulation, chords, counterpoint,
  and offline effects (with ring-out tail).
- Live visuals: a `gr.Timer` polls the live player ~2×/s for an
  oscilloscope, a scrolling 3 s spectrogram, and a digit ticker.

### Verification
- All 14 modes × 20 constants (280 combos) generate finite stereo audio.
- All modes/waveforms run through the live callback with
  `last_callback_error = None`.
- Effects: echo spacing exact; block-streaming == offline (0 max error);
  reverb tail rings out.
- Preset save/load round-trips; classic UI still constructs against the
  refactored library.
