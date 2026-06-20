# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental audio synthesis project that explores sonification of mathematical constants, specifically the digits of π (pi). The project generates musical tones by mapping decimal digits to frequencies and plays them as audio sequences.

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
- Interactive UI: `./run_ui.sh` (or `/mnt/e/anaconda3/python.exe app.py`) — launches a local Gradio server (default `http://127.0.0.1:7860`). Controls are grouped into tabs — **Source** (constant, digits, duration, volume, pan), **Tuning** (14 frequency modes incl. scales, just intonation, Bohlen–Pierce, Pythagorean, golden-ratio, prime/inharmonic), **Timbre** (sine/saw/square/triangle/pulse waveforms, brightness, FM, crossfade-or-ADSR envelope, chords), **Modulation** (a second constant's digits steer rhythm/transpose/harmonic/dynamics/brightness/waveform/vibrato/pan per note, plus a simultaneous counterpoint voice), **FX** (chorus, delay, reverb), **Visuals** — with save/load presets (`presets.json`, gitignored). The UI offers two playback modes:
  - **Generate** — synthesizes a fixed stereo buffer, shows the spectrogram, and plays in the browser. Good for sharing or downloading a snippet.
  - **Start Live / Stop Live** (`live.py`) — opens a stereo `sounddevice.OutputStream` that plays continuously on the **host machine's speakers** while reading parameters from a shared dict. Slider changes are heard within ~50 ms with no restart-from-beginning. While running, a `gr.Timer` polls `LivePlayer.get_visual_snapshot()` to update an oscilloscope, a live spectrogram of the last 3 s, and a digit ticker. Local-only: audio plays where `app.py` runs, not in the browser, so this would not work over a network deployment.
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
./run-sandbox.sh --claude        # launch the Claude Code CLI inside the sandbox (enables network)
NETWORK=1 ./run-sandbox.sh ...   # enable networking without the UI helper
AUDIO=1   ./run-sandbox.sh ...   # enable host audio without the --audio flag
```

Isolation defaults: `--cap-drop ALL`, `--security-opt no-new-privileges`, `--network none`, and only this folder mounted (host drives under `/mnt` are not visible).

- **Audio** is opt-in. WSLg runs a PulseAudio server at `/mnt/wslg/PulseServer` that pipes to the Windows speakers. With `--audio`/`AUDIO=1` the launcher mounts that socket and sets `PULSE_SERVER`; the container's ALSA default routes through it. This is the way to get real playback (incl. `live.py`) from inside the sandbox. Cost: the container can then talk to the host's PulseAudio server, so it's off by default.
- **Network** is off by default; `--ui`, `--claude`, and `NETWORK=1` enable Docker's bridge network.
- **Claude Code** is baked into the image (native binary at `/root/.local/bin/claude`, no Node.js). Run it with `./run-sandbox.sh --claude` (which turns on networking, since it must reach `api.anthropic.com`). Authentication: if `ANTHROPIC_API_KEY` is exported, the launcher forwards it (network-only, same policy as the OpenAI key); otherwise it reads `~/.config/irrational/anthropic_api_key` if present. Override the path with `ANTHROPIC_API_KEY_FILE`. Without a key, run `claude` once interactively to log in — but note the container is `--rm`, so that login does not persist between runs.
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
- `synth.py` - Shared synthesis engine: `render_wave()` (sine/saw/square/triangle/pulse + brightness + FM, phase-based so live stays click-free), `adsr_envelope()`, `pan_gains()`, `render_sequence()` (offline note-event renderer: chords, per-note timbre, vibrato, stereo pan, crossfade/ADSR)
- `modulation.py` - Cross-modulation: `MOD_TARGETS` (digit → rhythm/transpose/harmonic/dynamics/brightness/waveform/vibrato/pan), `apply_modulation()`, `build_note_sequence()`
- `effects.py` - Streamable chorus / feedback delay / Schroeder reverb (`EffectChain` — block-split processing is bit-identical to whole-buffer; used by both paths)
- `live.py` - `LivePlayer` + `_Voice`: stereo real-time engine with full feature parity (modulation, counterpoint voice, chords, effects) and a rolling buffer for the live visuals
- `presets.py` - Save/load UI presets to `presets.json` (gitignored)
- `app.py` - New tabbed Gradio UI (port 7860); `app_classic.py` + `run_ui_classic.sh` - pre-revision UI kept for comparison (port 7861)
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