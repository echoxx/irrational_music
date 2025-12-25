import numpy as np
import sounddevice as sd
from mpmath import mp

# Optional: for visualization (install with: pip install matplotlib)
try:
    import matplotlib.pyplot as plt
    from scipy import signal
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# =============================================================================
# FREQUENCY MAPPING FUNCTIONS
# =============================================================================

def calculate_frequencies_equal_temperament(start_freq, num_steps=10, num_octaves=1, precision=2):
    """
    Calculate frequencies dividing octaves into equal steps (original method).
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


def calculate_frequencies_harmonic_series(base_freq=220, num_harmonics=10):
    """
    Calculate frequencies using the natural harmonic series.
    All frequencies are integer multiples of the fundamental, creating
    mathematically related tones that blend naturally.

    Parameters:
    base_freq (float): Fundamental frequency in Hz (default: 220 Hz = A3)
    num_harmonics (int): Number of harmonics to generate (default: 10 for digits 0-9)

    Returns:
    list: Frequencies for harmonics 1 through num_harmonics

    Harmonic series for base_freq=220:
    0 → 220 Hz  (1st harmonic, fundamental)
    1 → 440 Hz  (2nd harmonic, octave)
    2 → 660 Hz  (3rd harmonic, octave + fifth)
    3 → 880 Hz  (4th harmonic, 2 octaves)
    4 → 1100 Hz (5th harmonic, 2 oct + major 3rd)
    5 → 1320 Hz (6th harmonic, 2 oct + fifth)
    6 → 1540 Hz (7th harmonic, ~minor 7th)
    7 → 1760 Hz (8th harmonic, 3 octaves)
    8 → 1980 Hz (9th harmonic, 3 oct + major 2nd)
    9 → 2200 Hz (10th harmonic, 3 oct + major 3rd)
    """
    return [base_freq * (i + 1) for i in range(num_harmonics)]


def calculate_frequencies_continuous(min_freq=110, max_freq=880, num_values=100):
    """
    Calculate frequencies as a continuous gradient for digit pairs (00-99).
    Each value gets a unique frequency.

    Parameters:
    min_freq (float): Lowest frequency (for value 0)
    max_freq (float): Highest frequency (for value 99)
    num_values (int): Number of discrete frequency values (default: 100)

    Returns:
    list: Frequencies mapped linearly across the range
    """
    return [min_freq + (i / (num_values - 1)) * (max_freq - min_freq) for i in range(num_values)]


# =============================================================================
# IRRATIONAL NUMBER FUNCTIONS
# =============================================================================

def get_pi(n):
    """Returns the first n digits of pi (including the 3)."""
    mp.dps = n + 10
    pi_str = str(mp.pi)
    pi_digits = pi_str.replace(".", "")[:n+1]
    return [int(d) for d in pi_digits]


def get_e(n):
    """Returns the first n digits of e (including the 2)."""
    mp.dps = n + 10
    e_str = str(mp.e)
    e_digits = e_str.replace(".", "")[:n+1]
    return [int(d) for d in e_digits]


def get_sqrt2(n):
    """Returns the first n digits of √2 (square root of 2)."""
    mp.dps = n + 10
    sqrt2_str = str(mp.sqrt(2))
    sqrt2_digits = sqrt2_str.replace(".", "")[:n+1]
    return [int(d) for d in sqrt2_digits]


def get_twelfth_root_of_2(n):
    """
    Returns the first n digits of the 12th root of 2.
    This is the semitone ratio in equal temperament (~1.05946309).
    """
    mp.dps = n + 10
    root12_str = str(mp.root(2, 12))
    root12_digits = root12_str.replace(".", "")[:n+1]
    return [int(d) for d in root12_digits]


def get_phi(n):
    """Returns the first n digits of φ (golden ratio)."""
    mp.dps = n + 10
    phi_str = str(mp.phi)
    phi_digits = phi_str.replace(".", "")[:n+1]
    return [int(d) for d in phi_digits]


def get_sqrt3(n):
    """Returns the first n digits of √3."""
    mp.dps = n + 10
    sqrt3_str = str(mp.sqrt(3))
    sqrt3_digits = sqrt3_str.replace(".", "")[:n+1]
    return [int(d) for d in sqrt3_digits]


def get_ln2(n):
    """Returns the first n digits of ln(2) (natural log of 2)."""
    mp.dps = n + 10
    ln2_str = str(mp.ln(2))
    # ln(2) starts with 0.693..., so handle leading zero
    ln2_digits = ln2_str.replace(".", "")[:n+1]
    return [int(d) for d in ln2_digits]


# Mapping of constant names to functions
IRRATIONAL_CONSTANTS = {
    'pi': ('Pi', get_pi, '3.14159...'),
    'e': ('e (Euler\'s number)', get_e, '2.71828...'),
    'sqrt2': ('sqrt(2) (Square root of 2)', get_sqrt2, '1.41421...'),
    'root12_2': ('12th root of 2 (semitone ratio)', get_twelfth_root_of_2, '1.05946...'),
    'phi': ('Phi (Golden ratio)', get_phi, '1.61803...'),
    'sqrt3': ('sqrt(3) (Square root of 3)', get_sqrt3, '1.73205...'),
    'ln2': ('ln(2) (Natural log of 2)', get_ln2, '0.69314...'),
}


def get_irrational_digits(constant, n):
    """
    Returns the first n digits of the specified irrational constant.

    Parameters:
    constant (str): One of: 'pi', 'e', 'sqrt2', 'root12_2', 'phi', 'sqrt3', 'ln2'
    n (int): Number of digits desired

    Returns:
    list: List of integers representing each digit
    """
    constant = constant.lower()
    if constant in IRRATIONAL_CONSTANTS:
        _, func, _ = IRRATIONAL_CONSTANTS[constant]
        return func(n)
    else:
        available = ', '.join(IRRATIONAL_CONSTANTS.keys())
        raise ValueError(f"Unknown constant '{constant}'. Choose from: {available}")


def get_irrational_digit_pairs(constant, n):
    """
    Returns digit pairs (00-99) from the irrational constant.
    Groups consecutive digits into pairs for finer frequency resolution.

    Parameters:
    constant (str): The irrational constant name
    n (int): Number of digit pairs desired

    Returns:
    list: List of integers 0-99 representing digit pairs
    """
    # Get twice as many single digits as we need pairs
    digits = get_irrational_digits(constant, n * 2 + 1)

    # Group into pairs
    pairs = []
    for i in range(0, len(digits) - 1, 2):
        pair_value = digits[i] * 10 + digits[i + 1]
        pairs.append(pair_value)
        if len(pairs) >= n:
            break

    return pairs


# =============================================================================
# FREQUENCY MAPPING
# =============================================================================

def map_digits_to_frequencies(digits, frequencies):
    """
    Maps single digits (0-9) to frequencies.

    Parameters:
    digits (list): List of digits 0-9
    frequencies (list): List of 10 frequencies

    Returns:
    list: List of frequencies corresponding to each digit
    """
    return [frequencies[d] for d in digits]


def map_digit_pairs_to_frequencies(digit_pairs, frequencies):
    """
    Maps digit pairs (0-99) to frequencies.

    Parameters:
    digit_pairs (list): List of values 0-99
    frequencies (list): List of 100 frequencies

    Returns:
    list: List of frequencies corresponding to each digit pair
    """
    return [frequencies[dp] for dp in digit_pairs]


# =============================================================================
# AUDIO PLAYBACK
# =============================================================================

def generate_audio(frequencies, duration=0.2, amplitude=0.3, sample_rate=44100, crossfade=0.05):
    """
    Generate audio buffer from a sequence of frequencies.
    Returns the audio array without playing it.
    """
    if len(frequencies) == 0:
        return np.array([], dtype=np.float32)

    samples_per_note = int(sample_rate * duration)
    crossfade_samples = int(sample_rate * crossfade)
    crossfade_samples = min(crossfade_samples, samples_per_note // 2)

    total_samples = samples_per_note + (len(frequencies) - 1) * (samples_per_note - crossfade_samples)
    audio = np.zeros(total_samples, dtype=np.float32)

    fade_out = np.cos(np.linspace(0, np.pi/2, crossfade_samples)) ** 2
    fade_in = np.sin(np.linspace(0, np.pi/2, crossfade_samples)) ** 2

    current_pos = 0
    for i, freq in enumerate(frequencies):
        t = np.linspace(0, duration, samples_per_note, False)
        tone = amplitude * np.sin(2 * np.pi * freq * t)

        if i > 0:
            tone[:crossfade_samples] *= fade_in
            audio[current_pos:current_pos + crossfade_samples] *= fade_out

        audio[current_pos:current_pos + samples_per_note] += tone

        if i < len(frequencies) - 1:
            current_pos += samples_per_note - crossfade_samples
        else:
            current_pos += samples_per_note

    return audio


def play_audio(audio, sample_rate=44100):
    """Play an audio buffer."""
    sd.play(audio, sample_rate)
    sd.wait()


def play_frequencies(frequencies, duration=0.2, amplitude=0.3, sample_rate=44100, crossfade=0.05):
    """Generate and play a sequence of frequencies."""
    audio = generate_audio(frequencies, duration, amplitude, sample_rate, crossfade)
    play_audio(audio, sample_rate)
    return audio


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_spectrogram(audio, sample_rate=44100, title="Spectrogram", save_path=None):
    """
    Generate and display a spectrogram of the audio.

    Parameters:
    audio (np.array): Audio data
    sample_rate (int): Sample rate in Hz
    title (str): Plot title
    save_path (str): Optional path to save the figure
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install matplotlib and scipy:")
        print("  pip install matplotlib scipy")
        return

    plt.figure(figsize=(12, 4))

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024, noverlap=512)

    # Plot
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.colorbar(label='Power (dB)')
    plt.ylim(0, 3000)  # Focus on audible range

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram to: {save_path}")

    plt.show()


def plot_comparison(audio1, audio2, label1, label2, sample_rate=44100, save_path=None):
    """
    Plot spectrograms of two audio sequences side by side for comparison.
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install matplotlib and scipy:")
        print("  pip install matplotlib scipy")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for ax, audio, label in [(axes[0], audio1, label1), (axes[1], audio2, label2)]:
        f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024, noverlap=512)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Spectrogram: {label}')
        ax.set_ylim(0, 3000)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")

    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Choose frequency mapping mode:
    # 'equal_temperament' - Original 10-step octave division
    # 'harmonic_series'   - Natural harmonics (recommended for emergent tones)
    # 'continuous'        - 100 frequencies for digit pairs (00-99)
    frequency_mode = 'harmonic_series'

    # Choose irrational constants to compare (up to 5):
    # Options: 'pi', 'e', 'sqrt2', 'root12_2', 'phi', 'sqrt3', 'ln2'
    # Add or remove constants from this list as desired
    constants_to_play = ['pi', 'e', 'phi', 'sqrt2', 'ln2']



    # Playback settings
    num_digits = 100             # Number of digits (or digit pairs if using 'continuous')
    note_duration = 0.05        # Seconds per note (try 0.02-0.05 for emergent tones)
    crossfade_time = 0.01       # Crossfade overlap
    volume = 0.3                # Volume (0.0 to 1.0)
    pause_duration = 3.0        # Pause between sequences

    # Visualization
    show_spectrograms = True    # Set to True to display spectrograms

    # =========================================================================
    # SETUP
    # =========================================================================

    print("\n" + "=" * 60)
    print("IRRATIONAL NUMBER SONIFICATION")
    print("=" * 60)
    print(f"Frequency Mode: {frequency_mode}")
    print(f"Constants ({len(constants_to_play)}): {', '.join(constants_to_play)}")
    print(f"Digits: {num_digits}, Duration: {note_duration}s/note")
    print("=" * 60)

    # Generate frequencies based on mode
    if frequency_mode == 'harmonic_series':
        freqs = calculate_frequencies_harmonic_series(base_freq=220, num_harmonics=10)
        print(f"\nHarmonic series frequencies (base=220Hz):")
        for i, f in enumerate(freqs):
            print(f"  {i} -> {f:.0f} Hz (harmonic {i+1})")
        use_pairs = False
    elif frequency_mode == 'continuous':
        freqs = calculate_frequencies_continuous(min_freq=110, max_freq=880, num_values=100)
        print(f"\nContinuous frequency mapping: 110Hz (00) to 880Hz (99)")
        use_pairs = True
    else:  # equal_temperament
        freqs = calculate_frequencies_equal_temperament(start_freq=440, num_steps=10, num_octaves=1)
        print(f"\nEqual temperament frequencies (440Hz base):")
        for i, f in enumerate(freqs[:10]):
            print(f"  {i} -> {f:.0f} Hz")
        use_pairs = False

    # =========================================================================
    # GENERATE ALL AUDIO AND DISPLAY SPECTROGRAMS
    # =========================================================================

    # Use the configured constants
    all_audio = []
    all_names = []

    print("\n" + "=" * 60)
    print("GENERATING AUDIO FOR ALL CONSTANTS")
    print("=" * 60)

    for const in constants_to_play:
        name, _, approx = IRRATIONAL_CONSTANTS[const]
        all_names.append(name)

        print(f"\n{name} ({approx})")

        if use_pairs:
            digits = get_irrational_digit_pairs(const, num_digits)
            print(f"  First {num_digits} digit pairs: {digits[:20]}...")
        else:
            digits = get_irrational_digits(const, num_digits)
            print(f"  First {num_digits} digits: {digits}")

        if use_pairs:
            mapped_freqs = map_digit_pairs_to_frequencies(digits, freqs)
        else:
            mapped_freqs = map_digits_to_frequencies(digits, freqs)

        print(f"  Generating audio...")
        audio = generate_audio(mapped_freqs, duration=note_duration, amplitude=volume, crossfade=crossfade_time)
        all_audio.append(audio)

    # Display all spectrograms at once (non-blocking)
    if show_spectrograms and VISUALIZATION_AVAILABLE:
        print("\n" + "=" * 60)
        print("DISPLAYING SPECTROGRAMS")
        print("=" * 60)

        # Create a single figure with subplots stacked vertically (one per constant)
        num_constants = len(constants_to_play)
        fig_height = 3 * num_constants  # 3 inches per subplot
        fig, axes = plt.subplots(num_constants, 1, figsize=(14, fig_height))
        fig.suptitle('Irrational Number Spectrograms', fontsize=16, fontweight='bold')

        # Ensure axes is always a list (if only 1 constant, axes is not a list)
        if num_constants == 1:
            axes = [axes]

        for i, (audio, name, ax) in enumerate(zip(all_audio, all_names, axes)):
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(audio, 44100, nperseg=1024, noverlap=512)

            # Plot
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{name}')
            ax.set_ylim(0, 3000)

            # Add colorbar for each subplot
            plt.colorbar(im, ax=ax, label='Power (dB)')

            print(f"  Created spectrogram for {name}")

        plt.tight_layout()

        # Show all spectrograms without blocking
        plt.show(block=False)
        print("\nAll spectrograms displayed in one window!")
        print("Waiting 2 seconds before playback...\n")
        plt.pause(2)

    # =========================================================================
    # PLAY ALL CONSTANTS
    # =========================================================================

    for i, (const, name, audio) in enumerate(zip(constants_to_play, all_names, all_audio)):
        print("=" * 60)
        print(f"Playing: {name}")
        print("=" * 60)
        play_audio(audio)
        print(f"Done with {const}!")

        # Pause between constants (but not after the last one)
        if i < len(constants_to_play) - 1:
            print(f"\nPausing for {pause_duration} seconds...\n")
            sd.sleep(int(pause_duration * 1000))

    print("\n" + "=" * 60)
    print("All sequences complete!")
    print("=" * 60)
