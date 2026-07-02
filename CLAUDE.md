# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental audio synthesis project that explores sonification of mathematical constants, specifically the digits of π (pi). The project generates musical tones by mapping decimal digits to frequencies and plays them as audio sequences.

User-facing documentation of every feature and control (constants, tunings, timbre, modulation, FX, playback modes) lives in `GUIDE.md`; this file covers the developer/architecture side.

## Development Environment

This project uses Jupyter notebooks as the primary development environment. Dependencies are managed through pip and conda.

### Required Dependencies

Install the following packages:
```bash
pip install numpy sounddevice mpmath scipy matplotlib
```

For the Gradio web UI (`app.py`), additionally install:
```bash
pip install gradio
```

Note: The project uses Anaconda/Miniconda environment. Some SciPy version warnings may appear but don't affect functionality.

### Running

- CLI playback: `./run.sh` (or `python irrational.py`) — plays the configured constants in sequence and shows spectrograms.
- Interactive UI: `./run_ui.sh` (or `/mnt/e/anaconda3/python.exe app.py`) — launches a local Gradio server (default `http://127.0.0.1:7860`). Controls are grouped into tabs — **Source** (constant, digits, duration, volume, pan), **Tuning** (14 frequency modes incl. scales, just intonation, Bohlen–Pierce, Pythagorean, golden-ratio, prime/inharmonic), **Timbre** (sine/saw/square/triangle/pulse waveforms plus a continuous morph blend, brightness, FM with irrational-ratio presets π/φ/√2/e, digit-driven harmonics — a constant's digits set the amplitudes of harmonics 1–16, optionally sliding note-by-note; overrides waveform/morph/brightness — crossfade-or-ADSR envelope, chords), **Modulation** (a second constant's digits steer rhythm/transpose/harmonic/dynamics/brightness/waveform/morph/vibrato/pan per note, plus a simultaneous counterpoint voice), **FX** (chorus, delay, Freeverb-style reverb with room size/damping/stereo width/pre-delay), **Visuals** — with save/load presets (`presets.json`, gitignored). The UI offers three playback modes:
  - **Generate** — synthesizes a fixed stereo buffer, shows the spectrogram, and plays in the browser. Good for sharing or downloading a snippet.
  - **Start Live / Stop Live** (`live.py`) — opens a stereo `sounddevice.OutputStream` that plays continuously on the **host machine's speakers** while reading parameters from a shared dict. Slider changes are heard within ~50 ms with no restart-from-beginning. A **⟲ Restart (digit 1)** button rewinds the digit walk to the start while playing (`LivePlayer.restart_sequence()` sets a flag; the audio callback rewinds both voices on the audio thread, letting in-flight notes finish so it's click-free). While running, a `gr.Timer` polls `LivePlayer.get_visual_snapshot()` to update an oscilloscope, a live spectrogram of the last 3 s, and a digit ticker. Local-only: audio plays where `app.py` runs, not in the browser, so this would not work over a network deployment.
  - **Record / Stop Record** (`live.py` `LivePlayer.start_recording`/`stop_recording`) — captures a live performance. Self-contained: Record auto-starts the live engine if it isn't already running, so you hit Record, drag any controls (constant, tuning, FX, …), and the whole take is captured. The callback appends each post-FX stereo block to a buffer under the same lock as the visuals tap; concatenation/peak-normalization (only when the take clips) happen on the UI thread in `stop_recording()`, never in the audio callback. There's a 10-minute soft cap (~210 MB). Stopping leaves playback running. The finished take appears in a `gr.Audio` player (browser playback + download) **and** is auto-saved as a timestamped `performance_YYYYMMDD_HHMMSS.wav` in the project folder (`.wav` is gitignored) via `scipy.io.wavfile`. No new dependency — scipy is already required.
- Classic UI: `./run_ui_classic.sh` (`app_classic.py`, port `7861`) — the pre-revision single-page interface, kept runnable for A/B comparison with the new tabbed UI. Both can run at once.

  Note: WSL's `python3` does not have the project's dependencies — the project uses the Windows Anaconda Python at `/mnt/e/anaconda3/python.exe`.

### Docker sandbox (`./run-sandbox.sh`)

For an isolated environment that **only has access to this folder**, use the Docker sandbox instead of the host Python. The image (`Dockerfile`) is `python:3.11-slim` with the deps from `requirements.txt` plus the PortAudio→ALSA→PulseAudio bridge libs. The source is **not** baked in — only this directory is bind-mounted at `/workspace` at runtime.

```bash
./run-sandbox.sh                 # interactive bash shell; folder-only, no network, no audio
./run-sandbox.sh python irrational.py
./run-sandbox.sh --audio         # + host-speaker audio (see below)
./run-sandbox.sh --audio python irrational.py   # CLI playback with sound
./run-sandbox.sh --ui            # Gradio UI: enables network + port 7860 + audio
./run-sandbox.sh --claude        # Claude Code CLI in the sandbox (network on; latest build, Fable model, --dangerously-skip-permissions)
NETWORK=1 ./run-sandbox.sh ...   # enable networking without the UI helper
AUDIO=1   ./run-sandbox.sh ...   # enable host audio without the --audio flag
```

Isolation defaults: `--cap-drop ALL`, `--security-opt no-new-privileges`, `--network none`, and only this folder mounted (host drives under `/mnt` are not visible).

- **Audio** is opt-in. WSLg runs a PulseAudio server at `/mnt/wslg/PulseServer` that pipes to the Windows speakers. With `--audio`/`AUDIO=1` the launcher mounts that socket and sets `PULSE_SERVER`; the container's ALSA default routes through it. This is the way to get real playback (incl. `live.py`) from inside the sandbox. Cost: the container can then talk to the host's PulseAudio server, so it's off by default.
- **Network** is off by default; `--ui`, `--claude`, and `NETWORK=1` enable Docker's bridge network.
- **Claude Code** is baked into the image (native binary at `/root/.local/bin/claude`, `latest` release channel, re-fetched at most once per day via a build-arg cache bust). `./run-sandbox.sh --claude` turns on networking (it must reach `api.anthropic.com`) and by default launches `claude --model claude-fable-5 --dangerously-skip-permissions` — the container itself is the permission boundary (folder-only mount, no caps), and `IS_SANDBOX=1` is set so Claude Code allows the flag as root. Pass your own command after `--claude` to override the defaults. Authentication: if `ANTHROPIC_API_KEY` is exported, the launcher forwards it (network-only, same policy as the OpenAI key); otherwise it reads `~/.config/irrational/anthropic_api_key` if present (override with `ANTHROPIC_API_KEY_FILE`). Without a key, log in once interactively — the login persists across runs in `~/.config/irrational/claude-config`, mounted into the container as `CLAUDE_CONFIG_DIR`.
- Requires Docker available in WSL (`docker --version`).

## Core Architecture

### Main Components

1. **Mathematical Computation (`mpmath` integration)**
   - Uses `mpmath` library for high-precision calculation of π
   - `mp.dps` controls decimal precision (can be set to 1000+ digits)
   - Extracts individual digits for frequency mapping

2. **Frequency Generation**
   - `calculate_frequencies()` function divides octaves into equal temperament steps
   - Creates frequency arrays using exponential scaling: `base_frequency * (2^(i/num_steps))`
   - Default base frequency is 440 Hz (A4)

3. **Audio Synthesis**
   - `generate_audio()` creates sine waves for each frequency
   - Uses numpy for signal generation: `np.sin(2 * np.pi * frequency * t)`
   - Supports configurable duration and sample rate (default 44100 Hz)

4. **Audio Playback**
   - `play_frequencies()` function handles real-time audio playback
   - Uses `sounddevice` library for cross-platform audio output
   - Implements fade-in/fade-out to prevent audio clicking
   - Supports configurable note duration (0.03s to 1s+)

5. **Digit-to-Frequency Mapping**
   - `map_numbers_to_frequencies()` creates mapping dictionaries
   - Maps digits 0-9 to specific frequency indices
   - `get_pi()` extracts decimal places from mathematical constants

## Key Technical Patterns

### Audio Processing
- All audio processing uses numpy arrays with float32 precision
- Sine wave generation: `amplitude * np.sin(2 * np.pi * frequency * t)`
- Envelope shaping with cosine-based fades for smooth transitions
- Sample rate typically set to 44100 Hz for CD-quality audio

### Mathematical Precision
- Use `decimal.getcontext().prec = N` for arbitrary precision arithmetic
- `mpmath.mp.dps` for setting decimal places in mpmath calculations
- Both libraries support 1000+ digit precision for mathematical exploration

### Jupyter Integration
- Code is organized in notebook cells for iterative development
- Audio playback works with both `sounddevice.play()` and IPython `Audio()` widgets
- Cells can be run independently for testing individual components

## Common Operations

### Generate and Play Pi Sequence
```python
# Set precision and get pi digits
mp.dps = 100
pi_digits = get_pi(50)  # First 50 decimal places

# Map to frequencies
freq_mapping = map_numbers_to_frequencies(range(10), calculate_frequencies(440))
pi_frequencies = [freq_mapping[digit] for digit in pi_digits]

# Play the sequence
play_frequencies(pi_frequencies, duration=0.1)
```

### Create Custom Frequency Scales
```python
# Generate custom frequency array
frequencies = calculate_frequencies(
    start_freq=220,    # Lower base note
    num_steps=12,      # 12-tone equal temperament
    num_octaves=2,     # Two octaves
    precision=2        # Round to 2 decimal places
)
```

### Adjust Audio Parameters
```python
play_frequencies(
    frequencies,
    duration=0.05,     # Faster playback
    amplitude=0.3,     # Quieter volume
    sample_rate=22050  # Lower quality for faster processing
)
```

## File Structure

- `irrational.py` - Core library: `IRRATIONAL_CONSTANTS` registry (20 constants incl. γ, Catalan, Apéry, Khinchin, e^π, Champernowne), `FREQUENCY_MODES` registry + `build_frequency_table()` (14 tuning modes), digit getters, `generate_audio()`, shared dark spectrogram helpers (`draw_spectrogram`, `style_dark_figure`), CLI `__main__`
- `synth.py` - Shared synthesis engine: `render_wave()` (sine/saw/square/triangle/pulse + continuous `morph` blend through them + brightness + FM incl. irrational-ratio presets (`FM_RATIO_PRESETS`, `resolve_fm_ratio()`) + `harmonics` additive mode fed by `harmonic_amps()` (digits → partial amplitudes; overrides waveform/morph/brightness); phase-based so live stays click-free), `adsr_envelope()`, `pan_gains()`, `render_sequence()` (offline note-event renderer: chords, per-note timbre, vibrato, stereo pan, crossfade/ADSR)
- `modulation.py` - Cross-modulation: `MOD_TARGETS` (digit → rhythm/transpose/harmonic/dynamics/brightness/waveform/morph/vibrato/pan), `apply_modulation()`, `build_note_sequence()` (also applies the digit-harmonics window per note via `harm_*` kwargs)
- `effects.py` - Streamable chorus / feedback delay / Freeverb-style reverb (8 lowpass-damped combs + 4 allpasses per channel; room size, damping, stereo width, pre-delay as live-safe floats) (`EffectChain` — block-split processing is bit-identical to whole-buffer, verified by `verify_features.py`; used by both paths)
- `live.py` - `LivePlayer` + `_Voice`: stereo real-time engine with full feature parity (modulation, counterpoint voice, chords, effects) and a rolling buffer for the live visuals; also captures performance recordings (`start_recording`/`stop_recording`, tapping the post-FX block in the callback)
- `presets.py` - Save/load UI presets to `presets.json` (gitignored)
- `verify_features.py` - Regression checks (run with the Anaconda python): `synthesize()`/`PARAM_KEYS` signature lockstep, reverb block-split bit-identity, timbre stability/spectral checks, live-voice smoke test
- `app.py` - New tabbed Gradio UI (port 7860); `app_classic.py` + `run_ui_classic.sh` - pre-revision UI kept for comparison (port 7861)
- `GUIDE.md` - User guide: what every constant, tuning mode, and UI control means
- `irrational.ipynb` - Main development notebook with core functions
- `2024-12-18_irrational.ipynb` - Latest experimental version
- `example.py` - Simple OpenAI API test script (reads `OPENAI_API_KEY` from the environment)
- `output.wav` - Generated audio file output
- `Dockerfile`, `requirements.txt`, `.dockerignore`, `run-sandbox.sh` - Docker sandbox (see "Docker sandbox" above)

## Security Notes

- **OpenAI API key lives outside this folder.** It is stored at `~/.config/irrational/openai_api_key` (perms `600`), not in the project directory, so it is never on the Docker sandbox mount. Code reads it via the standard `OPENAI_API_KEY` environment variable (`OpenAI()` picks it up automatically); the sandbox launcher loads it from that file and injects it as an env var **only when networking is enabled**. Override the path with `OPENAI_API_KEY_FILE`, or just `export OPENAI_API_KEY=...` yourself (the launcher prefers an already-set env var).
- **Anthropic API key** (for Claude Code in the sandbox) follows the same pattern: stored at `~/.config/irrational/anthropic_api_key` (outside the mount), read into `ANTHROPIC_API_KEY` and forwarded only when networking is enabled. Override with `ANTHROPIC_API_KEY_FILE`.
- Never commit API keys to version control (`openai_api_key` is in `.gitignore`).
- Prefer environment variables / secure key management over hardcoded keys in notebook cells.