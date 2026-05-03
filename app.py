"""Gradio UI for the irrational sonification project.

Run with: python app.py
"""

import matplotlib
matplotlib.use("Agg")  # headless backend; Gradio renders the figure

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from irrational import (
    IRRATIONAL_CONSTANTS,
    calculate_frequencies_continuous,
    calculate_frequencies_equal_temperament,
    calculate_frequencies_harmonic_series,
    calculate_frequencies_microtonal,
    generate_audio,
    get_irrational_digit_pairs,
    get_irrational_digits,
    map_digit_pairs_to_frequencies,
    map_digits_to_frequencies,
)
from live import LivePlayer

SAMPLE_RATE = 44100

live_player = LivePlayer()

CONSTANT_CHOICES = [(name, key) for key, (name, _, _) in IRRATIONAL_CONSTANTS.items()]
MODE_CHOICES = [
    ("Harmonic series", "harmonic_series"),
    ("Equal temperament", "equal_temperament"),
    ("Continuous (digit pairs, 100 frequencies)", "continuous"),
    ("Microtonal (subdivisions of equal temperament)", "microtonal"),
]


def build_spectrogram(audio, sample_rate, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(audio) == 0:
        ax.set_title(f"{title} (no audio)")
        return fig
    f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024, noverlap=512)
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_ylim(0, 3000)
    fig.colorbar(im, ax=ax, label="Power (dB)")
    fig.tight_layout()
    return fig


def synthesize(constant, num_digits, mode, base_freq, subdivisions, duration, crossfade, volume):
    num_digits = int(num_digits)
    subdivisions = max(1, int(subdivisions))

    if mode == "harmonic_series":
        freqs = calculate_frequencies_harmonic_series(base_freq=base_freq, num_harmonics=10)
        digits = get_irrational_digits(constant, num_digits)
        mapped = map_digits_to_frequencies(digits, freqs)
    elif mode == "equal_temperament":
        freqs = calculate_frequencies_equal_temperament(start_freq=base_freq, num_steps=10, num_octaves=1)
        digits = get_irrational_digits(constant, num_digits)
        mapped = map_digits_to_frequencies(digits, freqs[:10])
    elif mode == "continuous":
        # base_freq sets the floor; span ~3 octaves above so digit pairs cover a wide range
        freqs = calculate_frequencies_continuous(min_freq=base_freq / 2, max_freq=base_freq * 4, num_values=100)
        digits = get_irrational_digit_pairs(constant, num_digits)
        mapped = map_digit_pairs_to_frequencies(digits, freqs)
    elif mode == "microtonal":
        freqs = calculate_frequencies_microtonal(start_freq=base_freq, subdivisions_per_step=subdivisions, num_notes=10)
        digits = get_irrational_digits(constant, num_digits)
        mapped = map_digits_to_frequencies(digits, freqs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    audio = generate_audio(
        mapped,
        duration=duration,
        amplitude=volume,
        sample_rate=SAMPLE_RATE,
        crossfade=min(crossfade, duration / 2),
    )

    name, _, _ = IRRATIONAL_CONSTANTS[constant]
    unit = "pairs" if mode == "continuous" else "digits"
    title = f"{name} — {mode}, {num_digits} {unit}"
    fig = build_spectrogram(audio, SAMPLE_RATE, title)

    return (SAMPLE_RATE, audio), fig


def update_subdivision_visibility(mode):
    return gr.update(visible=(mode == "microtonal"))


def start_live(constant, num_digits, mode, base_freq, subdivisions, duration, volume):
    live_player.set_params(
        constant=constant,
        num_digits=int(num_digits),
        mode=mode,
        base_freq=float(base_freq),
        subdivisions=int(subdivisions),
        duration=float(duration),
        volume=float(volume),
    )
    live_player.refresh_digits()
    live_player.start()
    return "Live: running"


def stop_live():
    live_player.stop()
    return "Live: stopped"


def live_set(key, refresh_digits=False):
    """Build a Gradio change-handler that pushes a single param into the live player."""
    def handler(value):
        if not live_player.is_running:
            return
        live_player.set_param(key, value)
        if refresh_digits:
            live_player.refresh_digits()
    return handler


with gr.Blocks(title="Irrational Sonification") as demo:
    gr.Markdown(
        "# Irrational number sonification\n"
        "Map the digits of π, e, φ, √2, and other constants to frequencies and listen."
    )

    with gr.Row():
        with gr.Column(scale=1):
            constant = gr.Dropdown(choices=CONSTANT_CHOICES, value="pi", label="Constant")
            num_digits = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Number of digits")
            mode = gr.Radio(choices=MODE_CHOICES, value="harmonic_series", label="Frequency mode")
            base_freq = gr.Slider(minimum=110, maximum=880, value=220, step=1, label="Base / tone (Hz)")
            subdivisions = gr.Slider(
                minimum=2, maximum=10, value=2, step=1,
                label="Microtonal subdivisions per semitone",
                visible=False,
            )
            duration = gr.Slider(minimum=0.01, maximum=0.5, value=0.05, step=0.005, label="Note duration (s)")
            crossfade = gr.Slider(minimum=0.0, maximum=0.05, value=0.01, step=0.001, label="Crossfade (s)")
            volume = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="Volume")
            btn = gr.Button("Generate", variant="primary")
            gr.Markdown(
                "### Live mode\n"
                "Plays continuously on the host machine's speakers. "
                "Drag any slider while running to hear changes immediately."
            )
            with gr.Row():
                start_btn = gr.Button("Start Live", variant="primary")
                stop_btn = gr.Button("Stop Live")
            live_status = gr.Markdown("Live: stopped")
        with gr.Column(scale=2):
            audio_out = gr.Audio(label="Audio", type="numpy")
            spec_out = gr.Plot(label="Spectrogram")

    mode.change(fn=update_subdivision_visibility, inputs=mode, outputs=subdivisions)
    btn.click(
        fn=synthesize,
        inputs=[constant, num_digits, mode, base_freq, subdivisions, duration, crossfade, volume],
        outputs=[audio_out, spec_out],
    )

    start_btn.click(
        fn=start_live,
        inputs=[constant, num_digits, mode, base_freq, subdivisions, duration, volume],
        outputs=live_status,
    )
    stop_btn.click(fn=stop_live, outputs=live_status)

    # Live-mode slider/control wiring. `change` fires on every drag tick — fine
    # because set_param is just a locked dict update. num_digits uses `release`
    # to avoid re-running mpmath on every intermediate value.
    constant.change(fn=live_set("constant", refresh_digits=True), inputs=constant)
    num_digits.release(fn=live_set("num_digits", refresh_digits=True), inputs=num_digits)
    mode.change(fn=live_set("mode", refresh_digits=True), inputs=mode)
    base_freq.change(fn=live_set("base_freq"), inputs=base_freq)
    subdivisions.change(fn=live_set("subdivisions"), inputs=subdivisions)
    duration.change(fn=live_set("duration"), inputs=duration)
    volume.change(fn=live_set("volume"), inputs=volume)


if __name__ == "__main__":
    demo.launch()
