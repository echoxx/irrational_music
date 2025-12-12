import numpy as np

import sounddevice as sd
from mpmath import mp


def calculate_frequencies(start_freq, num_steps=10, num_octaves=1, precision=2):
    """
    Calculate frequencies dividing octaves into equal steps.

    Parameters:
    start_freq (float): Starting frequency in Hz
    num_steps (int): Number of steps to divide each octave into (default: 10)
    num_octaves (float): Number of octaves to calculate (default: 1)
    precision (int): Number of decimal places for rounding (default: 2)

    Returns:
    list: List of frequencies including start and end frequencies
    """
    total_steps = int(num_steps * num_octaves)
    target_multiplier = 2 ** num_octaves
    step_factor = target_multiplier ** (1/total_steps)

    frequencies = []
    current_freq = start_freq

    for _ in range(total_steps + 1):
        frequencies.append(round(current_freq, precision))
        current_freq *= step_factor

    return frequencies


def get_pi(n):
    """
    Returns the first n decimal places of pi using mpmath for high precision.

    Parameters:
    n (int): Number of decimal places desired

    Returns:
    list: List of integers representing each decimal place of pi
    """
    mp.dps = n + 10  # Set precision with some buffer
    pi_str = str(mp.pi)

    # Remove the decimal point and take first n+1 digits (including the 3)
    pi_digits = pi_str.replace(".", "")[:n+1]

    return [int(d) for d in pi_digits]


def map_numbers_to_frequencies(numbers, frequencies):
    """
    Maps single digits to frequency indices.

    Parameters:
    numbers (list): List of integers 0-9
    frequencies (list): List of frequencies

    Returns:
    dict: Dictionary mapping each number to its corresponding frequency
    """
    if not all(0 <= num <= 9 for num in numbers):
        raise ValueError("All numbers must be between 0 and 9")

    mapping = {num: frequencies[idx] for idx, num in enumerate(numbers)}

    return mapping


def play_frequencies(frequencies, duration=0.2, amplitude=0.3, sample_rate=44100, crossfade=0.05):
    """
    Play a sequence of frequencies as sine waves with smooth crossfade transitions.

    Parameters:
    frequencies (list): List of frequencies in Hz to play
    duration (float): Duration of each tone in seconds
    amplitude (float): Volume of the tone (0.0 to 1.0)
    sample_rate (int): Audio sample rate in Hz
    crossfade (float): Crossfade duration in seconds between notes (default: 0.05s)
    """
    if len(frequencies) == 0:
        return

    # Calculate samples per note and crossfade
    samples_per_note = int(sample_rate * duration)
    crossfade_samples = int(sample_rate * crossfade)

    # Ensure crossfade isn't longer than the note
    crossfade_samples = min(crossfade_samples, samples_per_note // 2)

    # Calculate total length: each note contributes (duration - crossfade) except the first
    # First note: full duration, subsequent notes overlap by crossfade amount
    total_samples = samples_per_note + (len(frequencies) - 1) * (samples_per_note - crossfade_samples)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Create crossfade curves using cosine for smooth transitions
    fade_out = np.cos(np.linspace(0, np.pi/2, crossfade_samples)) ** 2
    fade_in = np.sin(np.linspace(0, np.pi/2, crossfade_samples)) ** 2

    # Generate and blend each note
    current_pos = 0
    for i, freq in enumerate(frequencies):
        # Create time array for this note
        t = np.linspace(0, duration, samples_per_note, False)

        # Generate sine wave
        tone = amplitude * np.sin(2 * np.pi * freq * t)

        # For all notes except the first, apply crossfade
        if i > 0:
            # Apply fade-in to beginning of current note
            tone[:crossfade_samples] *= fade_in
            # Apply fade-out to end of previous note (already in buffer)
            audio[current_pos:current_pos + crossfade_samples] *= fade_out

        # Add the current tone to the buffer
        audio[current_pos:current_pos + samples_per_note] += tone

        # Move position for next note (overlap by crossfade amount)
        if i < len(frequencies) - 1:
            current_pos += samples_per_note - crossfade_samples
        else:
            current_pos += samples_per_note

    # Play the entire sequence as one continuous audio stream
    sd.play(audio, sample_rate)
    sd.wait()


if __name__ == "__main__":
    # Generate frequency scale (10 steps from A4)
    freqs = calculate_frequencies(start_freq=440, num_steps=10, num_octaves=1, precision=2)

    # Get first 50 decimal places of pi
    pi_digits = get_pi(50)
    print(f"First 50 digits of pi: {pi_digits}")

    # Map digits 0-9 to frequencies
    freq_mapping = map_numbers_to_frequencies(list(range(10)), freqs)

    # Convert pi digits to frequencies
    pi_frequencies = [freq_mapping[digit] for digit in pi_digits]

    # Play the sequence with smooth crossfade transitions
    print("Playing pi sequence...")
    play_frequencies(pi_frequencies, duration=0.2, amplitude=0.3, crossfade=0.075)
    print("Done!")

