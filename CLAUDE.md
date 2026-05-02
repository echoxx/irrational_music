# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental audio synthesis project that explores sonification of mathematical constants, specifically the digits of π (pi). The project generates musical tones by mapping decimal digits to frequencies and plays them as audio sequences.

## Development Environment

This project uses Jupyter notebooks as the primary development environment. Dependencies are managed through pip and conda.

### Required Dependencies

Install the following packages:
```bash
pip install numpy sounddevice mpmath scipy
```

Note: The project uses Anaconda/Miniconda environment. Some SciPy version warnings may appear but don't affect functionality.

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

- `irrational.ipynb` - Main development notebook with core functions
- `2024-12-18_irrational.ipynb` - Latest experimental version
- `example.py` - Simple OpenAI API test script
- `output.wav` - Generated audio file output
- `openai_api_key` - API key file (keep secure)

## Security Notes

The repository contains an OpenAI API key file. When working with this project:
- Never commit API keys to version control
- Use environment variables or secure key management
- The existing API key in the notebook cells should be replaced with secure alternatives