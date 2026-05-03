"""Real-time live-playback mode for the irrational sonification UI.

LivePlayer owns a sounddevice.OutputStream whose audio callback synthesizes
each block on demand from a shared, lock-protected params dict. The Gradio UI
mutates the dict from the main thread; the audio thread reads a snapshot at
the top of each callback and produces phase-continuous sine waves so that
mid-stream parameter changes don't click.
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from irrational import (
    calculate_frequencies_continuous,
    calculate_frequencies_equal_temperament,
    calculate_frequencies_harmonic_series,
    calculate_frequencies_microtonal,
    get_irrational_digit_pairs,
    get_irrational_digits,
)

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048


def _mode_uses_pairs(mode: str) -> bool:
    return mode == "continuous"


def _build_freq_table(mode: str, base_freq: float, subdivisions: int) -> list[float]:
    if mode == "harmonic_series":
        return calculate_frequencies_harmonic_series(base_freq=base_freq, num_harmonics=10)
    if mode == "equal_temperament":
        return calculate_frequencies_equal_temperament(
            start_freq=base_freq, num_steps=10, num_octaves=1
        )[:10]
    if mode == "continuous":
        return calculate_frequencies_continuous(
            min_freq=base_freq / 2, max_freq=base_freq * 4, num_values=100
        )
    if mode == "microtonal":
        return calculate_frequencies_microtonal(
            start_freq=base_freq,
            subdivisions_per_step=max(1, subdivisions),
            num_notes=10,
        )
    raise ValueError(f"Unknown mode: {mode}")


class LivePlayer:
    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.lock = threading.Lock()

        self.params = {
            "constant": "pi",
            "num_digits": 100,
            "mode": "harmonic_series",
            "base_freq": 220.0,
            "subdivisions": 2,
            "duration": 0.05,
            "volume": 0.3,
        }

        self.digits: list[int] = []
        self.digits_key: tuple | None = None

        self._freq_table: list[float] = []
        self._freq_table_key: tuple | None = None

        self.phase = 0.0
        self.digit_index = 0
        self.samples_into_note = 0

        self.stream: sd.OutputStream | None = None
        self.last_callback_error: str | None = None

    # ------------------------------------------------------------------ public

    def set_param(self, key: str, value) -> None:
        with self.lock:
            self.params[key] = value

    def set_params(self, **kwargs) -> None:
        with self.lock:
            self.params.update(kwargs)

    def refresh_digits(self) -> None:
        """Re-fetch the digit sequence based on current constant/num_digits/mode.

        Call from the UI thread (it does mpmath work). The new list is swapped
        in atomically under the lock; the audio thread reads it on the next
        callback.
        """
        with self.lock:
            constant = self.params["constant"]
            num_digits = int(self.params["num_digits"])
            uses_pairs = _mode_uses_pairs(self.params["mode"])
        key = (constant, num_digits, uses_pairs)
        if key == self.digits_key:
            return
        if uses_pairs:
            new_digits = get_irrational_digit_pairs(constant, num_digits)
        else:
            new_digits = get_irrational_digits(constant, num_digits)
        with self.lock:
            self.digits = new_digits
            self.digits_key = key
            if self.digit_index >= len(new_digits):
                self.digit_index = 0

    def start(self) -> None:
        if self.stream is not None:
            return
        self.refresh_digits()
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None

    @property
    def is_running(self) -> bool:
        return self.stream is not None

    # --------------------------------------------------------------- internal

    def _get_freq_table(self, mode: str, base_freq: float, subdivisions: int) -> list[float]:
        key = (mode, round(base_freq, 4), int(subdivisions))
        if key != self._freq_table_key:
            self._freq_table = _build_freq_table(mode, base_freq, int(subdivisions))
            self._freq_table_key = key
        return self._freq_table

    def _current_freq(self, p: dict, digits: list[int]) -> float:
        if not digits:
            return 0.0
        table = self._get_freq_table(p["mode"], p["base_freq"], p["subdivisions"])
        digit = digits[self.digit_index % len(digits)]
        if _mode_uses_pairs(p["mode"]):
            return table[digit % len(table)]
        return table[digit % len(table)]

    def _callback(self, outdata, frames, time_info, status):
        try:
            with self.lock:
                p = dict(self.params)
                digits = self.digits

            if not digits:
                outdata.fill(0.0)
                return

            samples_per_note = max(1, int(self.sample_rate * p["duration"]))
            out = np.zeros(frames, dtype=np.float32)

            written = 0
            while written < frames:
                # If we've overshot the (possibly newly-reduced) note length,
                # advance to the next digit BEFORE computing chunk so chunk
                # is always >= 1 and the loop makes forward progress.
                if self.samples_into_note >= samples_per_note:
                    self.samples_into_note = 0
                    self.digit_index = (self.digit_index + 1) % len(digits)

                freq = self._current_freq(p, digits)
                remaining_in_note = samples_per_note - self.samples_into_note
                chunk = min(frames - written, remaining_in_note)
                if chunk <= 0:
                    break  # defensive — should not happen after the guard above

                omega = 2.0 * np.pi * float(freq) / self.sample_rate
                idx = np.arange(chunk, dtype=np.float64)
                out[written:written + chunk] = (
                    float(p["volume"]) * np.sin(self.phase + omega * idx)
                ).astype(np.float32)
                self.phase = (self.phase + omega * chunk) % (2.0 * np.pi)
                self.samples_into_note += chunk
                written += chunk

            outdata[:, 0] = out
        except Exception:
            # Never let an exception escape into cffi. Fill silence and
            # remember it for diagnostics.
            outdata.fill(0.0)
            import traceback
            self.last_callback_error = traceback.format_exc()
