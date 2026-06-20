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


def calculate_frequencies_microtonal(start_freq=220, subdivisions_per_step=2, num_notes=10):
    """
    Calculate microtonal frequencies by subdividing each equal-temperament semitone.
    subdivisions_per_step=2 → quarter-tones (24 steps/octave),
    =4 → eighth-tones (48 steps/octave), etc.

    Parameters:
    start_freq (float): Starting frequency in Hz
    subdivisions_per_step (int): Microtonal steps per semitone (≥1, where 1 = standard 12-tet)
    num_notes (int): Number of frequencies to generate (default: 10 for digits 0-9)

    Returns:
    list: Successive microtonal frequencies starting at start_freq
    """
    total_steps_per_octave = 12 * subdivisions_per_step
    step_factor = 2 ** (1.0 / total_steps_per_octave)
    return [start_freq * (step_factor ** i) for i in range(num_notes)]


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


# Scale interval patterns in semitones from the tonic.
SCALE_INTERVALS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'major_pentatonic': [0, 2, 4, 7, 9],
    'chromatic': list(range(12)),
}


def calculate_frequencies_scale(start_freq=220, scale='major', num_notes=10):
    """
    Map digits onto the degrees of a familiar musical scale (12-TET).
    Degrees beyond one octave wrap upward into the next octave, so 10 digits
    span a bit more than an octave of the chosen scale.

    Parameters:
    start_freq (float): Tonic frequency in Hz
    scale (str): One of SCALE_INTERVALS: 'major', 'minor', 'major_pentatonic', 'chromatic'
    num_notes (int): Number of frequencies to generate (default: 10 for digits 0-9)

    Returns:
    list: Frequencies for successive scale degrees
    """
    intervals = SCALE_INTERVALS[scale]
    freqs = []
    for i in range(num_notes):
        octave, degree = divmod(i, len(intervals))
        semitones = intervals[degree] + 12 * octave
        freqs.append(start_freq * 2 ** (semitones / 12))
    return freqs


def calculate_frequencies_just(start_freq=220, num_notes=10):
    """
    Just intonation: pure small-integer ratios above the tonic (5-limit).
    Maximally consonant, beat-free intervals — a contrast to equal temperament.

    Ratios: 1/1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2/1, 9/4, 5/2
    """
    ratios = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2, 9/4, 5/2]
    return [start_freq * r for r in ratios[:num_notes]]


def calculate_frequencies_bohlen_pierce(start_freq=220, num_notes=10):
    """
    Bohlen–Pierce scale: 13 equal divisions of the 3:1 'tritave' instead of
    the 2:1 octave. Sounds alien yet internally consistent.
    """
    return [start_freq * 3 ** (i / 13) for i in range(num_notes)]


def calculate_frequencies_pythagorean(start_freq=220, num_notes=10):
    """
    Pythagorean tuning: stack pure 3:2 fifths and fold them back into a single
    octave, then sort ascending. Pure fifths, characteristically wide thirds.
    """
    ratios = []
    for k in range(num_notes):
        r = (3 / 2) ** k
        while r >= 2:
            r /= 2
        ratios.append(r)
    return [start_freq * r for r in sorted(ratios)]


def calculate_frequencies_golden(start_freq=220, num_notes=10):
    """
    Golden-ratio tuning: successive powers of φ folded into one octave and
    sorted. Because φ is irrational the pitches never coincide with any equal
    temperament — a shimmering, never-repeating scale.
    """
    phi = (1 + 5 ** 0.5) / 2
    ratios = []
    for k in range(num_notes):
        r = phi ** k
        while r >= 2:
            r /= 2
        ratios.append(r)
    return [start_freq * r for r in sorted(ratios)]


def calculate_frequencies_prime_harmonic(base_freq=110, num_notes=10):
    """
    Prime harmonics: only the prime-numbered partials of the fundamental
    (2, 3, 5, 7, 11, ...). Skipping composite harmonics gives a hollow,
    bell-like spectral character.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    return [base_freq * p for p in primes[:num_notes]]


def calculate_frequencies_inharmonic(base_freq=110, stretch=1.3, num_notes=10):
    """
    Stretched (inharmonic) partials: base * n^stretch. With stretch > 1 the
    partials spread wider than the harmonic series, like a struck bell or
    metal bar.

    Parameters:
    base_freq (float): Frequency of the first partial
    stretch (float): Exponent (1.0 = harmonic series; ~1.3 = bell-like)
    num_notes (int): Number of partials to generate
    """
    return [base_freq * (n + 1) ** stretch for n in range(num_notes)]


# =============================================================================
# FREQUENCY MODE REGISTRY
# =============================================================================
# Single source of truth for every frequency mode: UI label, table builder,
# and whether the mode consumes digit pairs (00-99) instead of single digits.
# Both the Generate path (app.py) and the live path (live.py) dispatch through
# build_frequency_table(), so adding a mode here makes it available everywhere.

FREQUENCY_MODES = {
    'harmonic_series': (
        'Harmonic series',
        lambda base, sub: calculate_frequencies_harmonic_series(base_freq=base, num_harmonics=10),
        False,
    ),
    'equal_temperament': (
        'Equal temperament',
        lambda base, sub: calculate_frequencies_equal_temperament(start_freq=base, num_steps=10, num_octaves=1)[:10],
        False,
    ),
    'continuous': (
        'Continuous (digit pairs, 100 frequencies)',
        lambda base, sub: calculate_frequencies_continuous(min_freq=base / 2, max_freq=base * 4, num_values=100),
        True,
    ),
    'microtonal': (
        'Microtonal (subdivisions of equal temperament)',
        lambda base, sub: calculate_frequencies_microtonal(start_freq=base, subdivisions_per_step=max(1, sub), num_notes=10),
        False,
    ),
    'scale_major': (
        'Major scale',
        lambda base, sub: calculate_frequencies_scale(start_freq=base, scale='major'),
        False,
    ),
    'scale_minor': (
        'Minor scale',
        lambda base, sub: calculate_frequencies_scale(start_freq=base, scale='minor'),
        False,
    ),
    'scale_pentatonic': (
        'Major pentatonic scale',
        lambda base, sub: calculate_frequencies_scale(start_freq=base, scale='major_pentatonic'),
        False,
    ),
    'scale_chromatic': (
        'Chromatic (12-TET)',
        lambda base, sub: calculate_frequencies_scale(start_freq=base, scale='chromatic'),
        False,
    ),
    'just': (
        'Just intonation (pure ratios)',
        lambda base, sub: calculate_frequencies_just(start_freq=base),
        False,
    ),
    'bohlen_pierce': (
        'Bohlen–Pierce (13 steps of 3:1)',
        lambda base, sub: calculate_frequencies_bohlen_pierce(start_freq=base),
        False,
    ),
    'pythagorean': (
        'Pythagorean (stacked fifths)',
        lambda base, sub: calculate_frequencies_pythagorean(start_freq=base),
        False,
    ),
    'golden': (
        'Golden-ratio tuning (powers of phi)',
        lambda base, sub: calculate_frequencies_golden(start_freq=base),
        False,
    ),
    'prime_harmonic': (
        'Prime harmonics (2,3,5,7,11,...)',
        lambda base, sub: calculate_frequencies_prime_harmonic(base_freq=base),
        False,
    ),
    'inharmonic': (
        'Inharmonic / bell (stretched partials)',
        lambda base, sub: calculate_frequencies_inharmonic(base_freq=base),
        False,
    ),
}


def build_frequency_table(mode, base_freq, subdivisions=2):
    """Build the digit→frequency table for a mode via FREQUENCY_MODES."""
    if mode not in FREQUENCY_MODES:
        available = ', '.join(FREQUENCY_MODES.keys())
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {available}")
    _, builder, _ = FREQUENCY_MODES[mode]
    return builder(float(base_freq), int(subdivisions))


def mode_uses_pairs(mode):
    """True if the mode consumes digit pairs (00-99) instead of single digits."""
    return FREQUENCY_MODES[mode][2]


# =============================================================================
# IRRATIONAL NUMBER FUNCTIONS
# =============================================================================

def _mp_digits(compute, n):
    """
    Shared digit extraction for mpmath-computed constants.
    Evaluates compute() at n+10 decimal places and returns the first n+1
    digits (integer part included, decimal point stripped) as ints.
    """
    mp.dps = n + 10
    digits_str = str(compute()).replace(".", "")[:n+1]
    return [int(d) for d in digits_str]


def get_pi(n):
    """Returns the first n digits of pi (including the 3)."""
    return _mp_digits(lambda: mp.pi, n)


def get_e(n):
    """Returns the first n digits of e (including the 2)."""
    return _mp_digits(lambda: mp.e, n)


def get_sqrt2(n):
    """Returns the first n digits of √2 (square root of 2)."""
    return _mp_digits(lambda: mp.sqrt(2), n)


def get_twelfth_root_of_2(n):
    """
    Returns the first n digits of the 12th root of 2.
    This is the semitone ratio in equal temperament (~1.05946309).
    """
    return _mp_digits(lambda: mp.root(2, 12), n)


def get_phi(n):
    """Returns the first n digits of φ (golden ratio)."""
    return _mp_digits(lambda: mp.phi, n)


def get_sqrt3(n):
    """Returns the first n digits of √3."""
    return _mp_digits(lambda: mp.sqrt(3), n)


def get_ln2(n):
    """Returns the first n digits of ln(2) (natural log of 2)."""
    return _mp_digits(lambda: mp.ln(2), n)


def get_sqrt5(n):
    """Returns the first n digits of √5."""
    return _mp_digits(lambda: mp.sqrt(5), n)


def get_sqrt7(n):
    """Returns the first n digits of √7."""
    return _mp_digits(lambda: mp.sqrt(7), n)


def get_cbrt2(n):
    """Returns the first n digits of ∛2 (cube root of 2)."""
    return _mp_digits(lambda: mp.root(2, 3), n)


def get_ln10(n):
    """Returns the first n digits of ln(10)."""
    return _mp_digits(lambda: mp.ln(10), n)


def get_euler_gamma(n):
    """Returns the first n digits of γ (Euler–Mascheroni constant, 0.5772...)."""
    return _mp_digits(lambda: mp.euler, n)


def get_catalan(n):
    """Returns the first n digits of Catalan's constant (0.9159...)."""
    return _mp_digits(lambda: mp.catalan, n)


def get_apery(n):
    """Returns the first n digits of Apéry's constant ζ(3) (1.2020...)."""
    return _mp_digits(lambda: mp.apery, n)


def get_khinchin(n):
    """Returns the first n digits of Khinchin's constant (2.6854...)."""
    return _mp_digits(lambda: mp.khinchin, n)


def get_gelfond(n):
    """Returns the first n digits of Gelfond's constant e^π (23.1406...)."""
    return _mp_digits(lambda: mp.e ** mp.pi, n)


def get_pi_to_pi(n):
    """Returns the first n digits of π^π (36.4621...)."""
    return _mp_digits(lambda: mp.pi ** mp.pi, n)


def get_silver_ratio(n):
    """Returns the first n digits of the silver ratio 1+√2 (2.4142...)."""
    return _mp_digits(lambda: 1 + mp.sqrt(2), n)


def get_golden_angle(n):
    """Returns the first n digits of the golden angle 360/φ² ≈ 137.5077... degrees."""
    return _mp_digits(lambda: 360 / mp.phi ** 2, n)


def get_champernowne(n):
    """
    Returns the first n digits of Champernowne's constant 0.123456789101112...
    Its decimal digits literally count upward, so it sounds like an
    ascending ramp — a clear audible contrast to 'random' digits.
    """
    digits = [0]  # the integer part, matching the other getters' convention
    i = 1
    while len(digits) < n + 1:
        digits.extend(int(d) for d in str(i))
        i += 1
    return digits[:n+1]


# Mapping of constant names to functions
IRRATIONAL_CONSTANTS = {
    'pi': ('Pi', get_pi, '3.14159...'),
    'e': ('e (Euler\'s number)', get_e, '2.71828...'),
    'sqrt2': ('sqrt(2) (Square root of 2)', get_sqrt2, '1.41421...'),
    'root12_2': ('12th root of 2 (semitone ratio)', get_twelfth_root_of_2, '1.05946...'),
    'phi': ('Phi (Golden ratio)', get_phi, '1.61803...'),
    'sqrt3': ('sqrt(3) (Square root of 3)', get_sqrt3, '1.73205...'),
    'ln2': ('ln(2) (Natural log of 2)', get_ln2, '0.69314...'),
    'sqrt5': ('sqrt(5) (Square root of 5)', get_sqrt5, '2.23606...'),
    'sqrt7': ('sqrt(7) (Square root of 7)', get_sqrt7, '2.64575...'),
    'cbrt2': ('cbrt(2) (Cube root of 2)', get_cbrt2, '1.25992...'),
    'ln10': ('ln(10) (Natural log of 10)', get_ln10, '2.30258...'),
    'euler_gamma': ('Gamma (Euler–Mascheroni)', get_euler_gamma, '0.57721...'),
    'catalan': ('Catalan\'s constant', get_catalan, '0.91596...'),
    'apery': ('Apery\'s constant zeta(3)', get_apery, '1.20205...'),
    'khinchin': ('Khinchin\'s constant', get_khinchin, '2.68545...'),
    'gelfond': ('e^pi (Gelfond\'s constant)', get_gelfond, '23.1406...'),
    'pi_pi': ('pi^pi', get_pi_to_pi, '36.4621...'),
    'silver': ('Silver ratio (1+sqrt(2))', get_silver_ratio, '2.41421...'),
    'golden_angle': ('Golden angle (360/phi^2)', get_golden_angle, '137.507...'),
    'champernowne': ('Champernowne (0.1234567891011...)', get_champernowne, '0.12345...'),
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

def generate_audio(frequencies, duration=0.2, amplitude=0.3, sample_rate=44100, crossfade=0.05,
                   waveform='sine', pulse_width=0.3, brightness=0.0, fm_depth=0.0, fm_ratio=2.0,
                   envelope='crossfade', adsr=None):
    """
    Generate audio buffer from a sequence of frequencies.
    Returns the audio array without playing it.

    Timbre is controlled by the optional kwargs (see synth.render_wave):
    waveform ('sine'/'sawtooth'/'square'/'triangle'/'pulse'), pulse_width,
    brightness (additive harmonics), fm_depth/fm_ratio (FM synthesis).
    envelope='adsr' shapes each note with `adsr` (attack, decay, sustain,
    release) instead of the classic crossfade overlap.
    """
    from synth import render_sequence

    notes = [
        {
            "freqs": freq,
            "duration": duration,
            "volume": amplitude,
            "waveform": waveform,
            "pulse_width": pulse_width,
            "brightness": brightness,
            "fm_depth": fm_depth,
            "fm_ratio": fm_ratio,
        }
        for freq in frequencies
    ]
    return render_sequence(
        notes,
        sample_rate=sample_rate,
        envelope=envelope,
        crossfade=crossfade,
        adsr=adsr,
        stereo=False,
    )


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

# Shared dark style for all spectrogram/oscilloscope figures.
DARK_BG = "#0e0e14"
DARK_FG = "#c8c8d4"
SPECTROGRAM_CMAP = "magma"


def style_dark_axis(ax):
    """Apply the shared dark theme to a matplotlib axis."""
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color("#3a3a4a")
    ax.tick_params(colors=DARK_FG, labelsize=8)
    ax.xaxis.label.set_color(DARK_FG)
    ax.yaxis.label.set_color(DARK_FG)
    ax.title.set_color("#eeeef6")


def draw_spectrogram(ax, audio, sample_rate=44100, title=None, fmax=3000,
                     log_freq=False, floor_db=-80):
    """
    Draw a styled spectrogram onto an existing axis (shared by the CLI plots
    and the Gradio UI so they match). Returns the mesh for an optional
    colorbar, or None if the audio is empty.

    Stereo input is mixed to mono for analysis. Power is normalized so the
    peak is 0 dB and the floor is floor_db, giving a consistent dynamic
    range regardless of volume. log_freq=True switches to a log frequency
    axis (better matches pitch perception).
    """
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        style_dark_axis(ax)
        if title:
            ax.set_title(f"{title} (no audio)")
        return None

    f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024, noverlap=512)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    Sxx_db -= Sxx_db.max()  # 0 dB = loudest bin

    mesh = ax.pcolormesh(t, f, Sxx_db, shading="gouraud",
                         cmap=SPECTROGRAM_CMAP, vmin=floor_db, vmax=0)
    style_dark_axis(ax)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    if title:
        ax.set_title(title)
    if log_freq:
        ax.set_yscale("log")
        ax.set_ylim(max(40.0, float(f[1])), fmax)
    else:
        ax.set_ylim(0, fmax)
    ax.grid(True, color="#2a2a38", linewidth=0.4, alpha=0.6)
    return mesh


def style_dark_figure(fig):
    """Apply the shared dark background to a figure (axes styled separately)."""
    fig.patch.set_facecolor(DARK_BG)


def add_spectrogram_colorbar(fig, mesh, ax):
    """Add a theme-matched colorbar for a spectrogram mesh."""
    cbar = fig.colorbar(mesh, ax=ax, label="Power (dB)")
    cbar.ax.yaxis.label.set_color(DARK_FG)
    cbar.ax.tick_params(colors=DARK_FG, labelsize=8)
    cbar.outline.set_edgecolor("#3a3a4a")
    return cbar


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

    fig, ax = plt.subplots(figsize=(12, 4))
    style_dark_figure(fig)
    mesh = draw_spectrogram(ax, audio, sample_rate, title=title)
    if mesh is not None:
        add_spectrogram_colorbar(fig, mesh, ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
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
    style_dark_figure(fig)

    for ax, audio, label in [(axes[0], audio1, label1), (axes[1], audio2, label2)]:
        draw_spectrogram(ax, audio, sample_rate, title=f'Spectrogram: {label}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
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
        style_dark_figure(fig)
        fig.suptitle('Irrational Number Spectrograms', fontsize=16, fontweight='bold',
                     color='#eeeef6')

        # Ensure axes is always a list (if only 1 constant, axes is not a list)
        if num_constants == 1:
            axes = [axes]

        for i, (audio, name, ax) in enumerate(zip(all_audio, all_names, axes)):
            mesh = draw_spectrogram(ax, audio, 44100, title=name)
            if mesh is not None:
                add_spectrogram_colorbar(fig, mesh, ax)

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
