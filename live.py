"""Real-time live-playback mode for the irrational sonification UI.

LivePlayer owns a sounddevice.OutputStream whose audio callback synthesizes
each stereo block on demand from a shared, lock-protected params dict. The
Gradio UI mutates the dict from the main thread; the audio thread reads a
snapshot at the top of each callback.

Feature parity with the Generate path: waveforms/brightness/FM (synth.py),
ADSR or micro-fade envelopes, chords, cross-modulation by a second constant
(modulation.py), an independent counterpoint voice, and streaming effects
(effects.py). Oscillator phases persist across note boundaries per chord
voice, so parameter changes and note transitions stay click-free; every note
additionally gets a short envelope (full ADSR, or a ~2 ms micro-fade in
crossfade mode) because non-sine waveforms aren't phase-aligned at note
boundaries.

For the live visuals, the callback also appends each rendered block to a
rolling ~3 s buffer and records which carrier/modulator/counterpoint digits
are currently sounding; the UI polls get_visual_snapshot() on a timer.
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from effects import EffectChain
from irrational import (
    build_frequency_table,
    get_irrational_digit_pairs,
    get_irrational_digits,
    mode_uses_pairs,
)
from modulation import apply_modulation
from synth import (NUM_HARMONICS, adsr_envelope, harmonic_amps, pan_gains,
                   render_wave, resolve_fm_ratio)

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
VISUAL_SECONDS = 3.0
MICRO_FADE_S = 0.002  # crossfade-mode per-note fade to avoid waveform clicks

# Kept as module-level aliases so existing imports/callers keep working.
_mode_uses_pairs = mode_uses_pairs
_build_freq_table = build_frequency_table


class _Voice:
    """One sequenced voice: walks a digit stream, one enveloped note at a time.

    Oscillator phases are kept per chord-slot and never reset, so frequency
    changes (new notes, live slider moves) are phase-continuous.
    """

    MAX_CHORD = 8

    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        self.phases = np.zeros(self.MAX_CHORD, dtype=np.float64)
        self.digit_index = 0
        self.sample_in_note = 0
        self.note: dict | None = None

    @staticmethod
    def _seq_index(i, n, loop_mode):
        """Map the running note counter onto a digit index.

        'forward': wrap straight back to the start (i mod n).
        'pingpong': bounce forward then backward (period 2n-2), so the
        sequence never jumps — the loop seam becomes inaudible.
        """
        if loop_mode == "pingpong" and n > 1:
            period = 2 * n - 2
            j = i % period
            return j if j < n else period - j
        return i % n

    def _advance_note(self, digits, mod_digits, harm_digits, vp, freq_table):
        loop_mode = vp.get("loop_mode", "forward")
        digit = digits[self._seq_index(self.digit_index, len(digits), loop_mode)]
        idx = digit % len(freq_table)
        chord_size = max(1, int(vp["chord_size"]))
        if chord_size > 1:
            step = int(vp["chord_step"])
            freqs = [freq_table[(idx + k * step) % len(freq_table)] for k in range(chord_size)]
        else:
            freqs = freq_table[idx]

        note = {
            "freqs": freqs,
            "duration": float(vp["duration"]),
            "volume": float(vp["volume"]),
            "pan": float(vp["pan"]),
            "waveform": vp["waveform"],
            "pulse_width": float(vp["pulse_width"]),
            "morph": float(vp["morph"]),
            "brightness": float(vp["brightness"]),
            "fm_depth": float(vp["fm_depth"]),
            "fm_ratio": float(vp["fm_ratio"]),
            "vibrato_depth": 0.0,
            "vibrato_rate": 5.0,
        }
        mod_digit = None
        if mod_digits and vp["mod_targets"]:
            mod_digit = mod_digits[self._seq_index(self.digit_index, len(mod_digits), loop_mode)]
            apply_modulation(note, vp["mod_targets"], mod_digit, vp["mod_depth"])

        if harm_digits:
            # digit window → additive partial amplitudes; built fresh per note
            # in the callback and never mutated afterwards (thread-safe).
            n = len(harm_digits)
            start = int(vp["harm_offset"])
            if vp["harm_slide"]:
                start += self._seq_index(self.digit_index, n, loop_mode)
            window = [harm_digits[(start + k) % n] for k in range(NUM_HARMONICS)]
            note["harmonics"] = harmonic_amps(window, vp["harm_rolloff"])

        samples = max(32, int(self.sr * note["duration"]))
        if vp["envelope"] == "adsr":
            env = adsr_envelope(samples, self.sr, *vp["adsr"])
        else:
            fade = min(samples // 4, max(8, int(MICRO_FADE_S * self.sr)))
            env = np.ones(samples, dtype=np.float32)
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            env[:fade] = ramp
            env[-fade:] = ramp[::-1]

        if not isinstance(note["freqs"], (list, tuple)):
            note["freqs"] = [note["freqs"]]
        gl, gr = pan_gains(note["pan"])
        note.update(samples=samples, env=env, gl=float(gl), gr=float(gr),
                    digit=int(digit), mod_digit=mod_digit)
        self.note = note
        self.sample_in_note = 0
        self.digit_index += 1

    def render(self, frames, digits, mod_digits, harm_digits, vp, freq_table):
        """Render `frames` samples; returns (frames, 2) float32."""
        out = np.zeros((frames, 2), dtype=np.float32)
        if not digits or not freq_table:
            return out

        written = 0
        while written < frames:
            if self.note is None or self.sample_in_note >= self.note["samples"]:
                self._advance_note(digits, mod_digits, harm_digits, vp, freq_table)
            note = self.note
            chunk = min(frames - written, note["samples"] - self.sample_in_note)

            seg = np.zeros(chunk, dtype=np.float64)
            vib_d, vib_r = note["vibrato_depth"], note["vibrato_rate"]
            for vi, freq in enumerate(note["freqs"][:self.MAX_CHORD]):
                if vib_d > 0.0:
                    t_abs = (self.sample_in_note + np.arange(chunk)) / self.sr
                    inst = freq * (1.0 + vib_d * np.sin(2.0 * np.pi * vib_r * t_abs))
                    incr = 2.0 * np.pi * inst / self.sr
                    ph = self.phases[vi] + np.concatenate(([0.0], np.cumsum(incr[:-1])))
                    self.phases[vi] = (ph[-1] + incr[-1]) % (2.0 * np.pi)
                else:
                    omega = 2.0 * np.pi * float(freq) / self.sr
                    ph = self.phases[vi] + omega * np.arange(chunk, dtype=np.float64)
                    self.phases[vi] = (self.phases[vi] + omega * chunk) % (2.0 * np.pi)
                seg += render_wave(ph, note["waveform"], note["pulse_width"],
                                   note["brightness"], note["fm_depth"], note["fm_ratio"],
                                   morph=note["morph"], harmonics=note.get("harmonics"))
            seg /= len(note["freqs"])

            env = note["env"][self.sample_in_note:self.sample_in_note + chunk]
            seg = (seg * env * note["volume"]).astype(np.float32)
            out[written:written + chunk, 0] += seg * note["gl"]
            out[written:written + chunk, 1] += seg * note["gr"]

            self.sample_in_note += chunk
            written += chunk
        return out


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
            "pan": 0.0,
            "loop_mode": "forward",  # or "pingpong" — seamless at the ends
            # timbre
            "waveform": "sine",
            "pulse_width": 0.3,
            "morph": 0.0,
            "brightness": 0.0,
            "fm_depth": 0.0,
            "fm_ratio": 2.0,
            "fm_preset": "custom",
            # digit-driven harmonics (constant key only — the digit list is
            # fetched in refresh_digits; no arrays ever live in params)
            "harm_constant": "none",
            "harm_slide": False,
            "harm_offset": 0,
            "harm_rolloff": 0.5,
            "envelope": "crossfade",
            "attack": 0.005,
            "decay": 0.04,
            "sustain": 0.7,
            "release": 0.04,
            # chords
            "chord_size": 1,
            "chord_step": 2,
            # modulation
            "mod_constant": "none",
            "mod_targets": (),
            "mod_depth": 0.5,
            # counterpoint
            "cp_constant": "none",
            "cp_mode": "harmonic_series",
            "cp_base_freq": 0.0,  # 0 → follow base_freq
            "cp_waveform": "sine",
            "cp_volume": 0.15,
            "cp_pan": 0.5,
            "cp_duration": 0.0,  # 0 → follow duration
            # effects
            "fx_chorus": 0.0,
            "fx_delay": 0.0,
            "fx_reverb": 0.0,
            "fx_room_size": 0.5,
            "fx_damping": 0.5,
            "fx_width": 1.0,
            "fx_predelay": 0.0,
        }

        self.digits: list[int] = []
        self.digits_key: tuple | None = None
        self.mod_digits: list[int] = []
        self.mod_digits_key: tuple | None = None
        self.cp_digits: list[int] = []
        self.cp_digits_key: tuple | None = None
        self.harm_digits: list[int] = []
        self.harm_digits_key: tuple | None = None

        self._freq_cache: dict[tuple, list[float]] = {}

        self.voice = _Voice(sample_rate)
        self.cp_voice = _Voice(sample_rate)
        self.effects = EffectChain(sample_rate, 2)

        self.visual_buffer = np.zeros((int(VISUAL_SECONDS * sample_rate), 2), dtype=np.float32)
        self.visual_pos = 0
        self.current_info: dict = {}

        # Performance recording: when active, the callback appends each rendered
        # post-FX block to _rec_blocks (under self.lock). Concatenation happens
        # on the UI thread in stop_recording(), never in the callback.
        self._recording = False
        self._rec_blocks: list[np.ndarray] = []
        self._rec_frames = 0
        self._rec_max_frames = int(self.sample_rate * 60 * 10)  # 10 min soft cap (~210 MB)
        self._rec_truncated = False

        # Set by restart_sequence() (UI thread), consumed by the audio
        # callback, which performs the actual rewind on the audio thread.
        self._restart_requested = False

        self.stream: sd.OutputStream | None = None
        self.last_callback_error: str | None = None

    # ------------------------------------------------------------------ public

    def set_param(self, key: str, value) -> None:
        with self.lock:
            self.params[key] = value

    def set_params(self, **kwargs) -> None:
        with self.lock:
            self.params.update(kwargs)

    def restart_sequence(self) -> None:
        """Rewind live playback to the first digit of the constant.

        Only sets a flag; the audio callback does the rewind on the audio
        thread (voice state is audio-thread-only). The notes currently
        sounding finish naturally — click-free — and the next note of each
        voice starts from digit index 0.
        """
        with self.lock:
            self._restart_requested = True

    def start_recording(self) -> None:
        """Begin capturing the live audio stream.

        Self-contained: starts the audio engine if it isn't already running, so
        Record works as a standalone mode. If Live is already running it just
        attaches to the existing stream with no gap.
        """
        if self.stream is None:
            self.start()
        with self.lock:
            self._rec_blocks = []
            self._rec_frames = 0
            self._rec_truncated = False
            self._recording = True  # set last: callback never sees a half-reset state

    def stop_recording(self):
        """Stop capturing and return (audio (N,2) float32, seconds, truncated).

        Returns None if nothing was captured. Playback is left running. The
        expensive concatenate/normalize run here on the UI thread (outside the
        lock) so the audio callback is never blocked.
        """
        with self.lock:
            self._recording = False
            blocks = self._rec_blocks
            self._rec_blocks = []
            truncated = self._rec_truncated
        if not blocks:
            return None
        audio = np.concatenate(blocks, axis=0)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1.0:
            audio = audio * (0.99 / peak)  # tame clipping; preserve relative dynamics
        return audio.astype(np.float32), len(audio) / self.sample_rate, truncated

    @property
    def is_recording(self) -> bool:
        return self._recording

    def recording_elapsed(self) -> float:
        with self.lock:
            return self._rec_frames / self.sample_rate

    def refresh_digits(self) -> None:
        """Re-fetch the carrier/modulator/counterpoint digit sequences.

        Call from the UI thread (it does mpmath work). New lists are swapped
        in atomically under the lock; the audio thread reads them on the next
        callback.
        """
        with self.lock:
            p = dict(self.params)
        num_digits = int(p["num_digits"])

        key = (p["constant"], num_digits, mode_uses_pairs(p["mode"]))
        if key != self.digits_key:
            new = (get_irrational_digit_pairs if key[2] else get_irrational_digits)(key[0], num_digits)
            with self.lock:
                self.digits = new
                self.digits_key = key

        # the modulator always uses single digits
        mkey = (p["mod_constant"], num_digits)
        if mkey != self.mod_digits_key:
            new = get_irrational_digits(mkey[0], num_digits) if mkey[0] != "none" else []
            with self.lock:
                self.mod_digits = new
                self.mod_digits_key = mkey

        # the harmonics spectrum stream also uses single digits
        hkey = (p["harm_constant"], num_digits)
        if hkey != self.harm_digits_key:
            new = get_irrational_digits(hkey[0], num_digits) if hkey[0] != "none" else []
            with self.lock:
                self.harm_digits = new
                self.harm_digits_key = hkey

        ckey = (p["cp_constant"], num_digits, p["cp_mode"] and mode_uses_pairs(p["cp_mode"]))
        if ckey != self.cp_digits_key:
            if ckey[0] != "none":
                new = (get_irrational_digit_pairs if ckey[2] else get_irrational_digits)(ckey[0], num_digits)
            else:
                new = []
            with self.lock:
                self.cp_digits = new
                self.cp_digits_key = ckey

    @staticmethod
    def _ensure_audio_device():
        """Make sure PortAudio has a usable output device; return one.

        PortAudio caches its device list when first initialized. In a
        long-running server process that list can be empty or stale (e.g.
        the process started before audio was available, or the Windows
        default device changed since), in which case the default output
        resolves to -1 and opening a stream fails. Re-initializing PortAudio
        rescans the hardware. Returns a device index to use explicitly, or
        None to use the (now valid) default.
        """
        def default_output():
            # sd.default.device is an (input, output) pair-like object
            try:
                dev = sd.default.device
                try:
                    out = dev[1]
                except (TypeError, IndexError):
                    out = dev
                return -1 if out is None else int(out)
            except Exception:
                return -1

        if default_output() >= 0:
            return None
        # rescan (safe here: this process has no open streams at this point)
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        if default_output() >= 0:
            return None
        # still no default — explicitly pick the first stereo-capable output
        try:
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_output_channels", 0) >= 2:
                    return i
        except Exception:
            pass
        return None  # let sounddevice raise its own descriptive error

    def start(self) -> None:
        if self.stream is not None:
            return
        self.refresh_digits()
        self.effects.reset()
        device = self._ensure_audio_device()
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            device=device,
            channels=2,
            dtype="float32",
            blocksize=self.block_size,
            # 'high' buys extra device-side buffering so brief GIL stalls
            # (UI visuals, mpmath refreshes) can't cause audible underruns.
            # Parameter changes still apply within ~0.1 s.
            latency="high",
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

    def get_visual_snapshot(self):
        """Return (recent_audio (N,2) oldest-first, info dict) for the UI."""
        with self.lock:
            pos = self.visual_pos
            buf = np.concatenate([self.visual_buffer[pos:], self.visual_buffer[:pos]])
            info = dict(self.current_info)
        return buf, info

    # --------------------------------------------------------------- internal

    def _get_freq_table(self, mode, base_freq, subdivisions):
        key = (mode, round(float(base_freq), 4), int(subdivisions))
        if key not in self._freq_cache:
            if len(self._freq_cache) > 256:
                self._freq_cache.clear()
            self._freq_cache[key] = build_frequency_table(mode, base_freq, int(subdivisions))
        return self._freq_cache[key]

    @staticmethod
    def _carrier_voice_params(p):
        return {
            "duration": p["duration"], "volume": p["volume"], "pan": p["pan"],
            "waveform": p["waveform"], "pulse_width": p["pulse_width"],
            "morph": p["morph"], "brightness": p["brightness"],
            "fm_depth": p["fm_depth"],
            "fm_ratio": resolve_fm_ratio(p["fm_preset"], p["fm_ratio"]),
            "harm_slide": p["harm_slide"], "harm_offset": p["harm_offset"],
            "harm_rolloff": p["harm_rolloff"],
            "envelope": p["envelope"],
            "adsr": (p["attack"], p["decay"], p["sustain"], p["release"]),
            "chord_size": p["chord_size"], "chord_step": p["chord_step"],
            "mod_targets": tuple(p["mod_targets"]), "mod_depth": p["mod_depth"],
            "loop_mode": p["loop_mode"],
        }

    @staticmethod
    def _cp_voice_params(p):
        return {
            "duration": p["cp_duration"] or p["duration"],
            "volume": p["cp_volume"], "pan": p["cp_pan"],
            "waveform": p["cp_waveform"], "pulse_width": p["pulse_width"],
            "morph": p["morph"], "brightness": p["brightness"],
            "fm_depth": p["fm_depth"],
            "fm_ratio": resolve_fm_ratio(p["fm_preset"], p["fm_ratio"]),
            # cp voice gets an empty harm stream; keys kept for vp uniformity
            "harm_slide": p["harm_slide"], "harm_offset": p["harm_offset"],
            "harm_rolloff": p["harm_rolloff"],
            "envelope": p["envelope"],
            "adsr": (p["attack"], p["decay"], p["sustain"], p["release"]),
            "chord_size": 1, "chord_step": 2,
            "mod_targets": (), "mod_depth": 0.0,
            "loop_mode": p["loop_mode"],
        }

    def _callback(self, outdata, frames, time_info, status):
        try:
            with self.lock:
                p = dict(self.params)
                digits = self.digits
                mod_digits = self.mod_digits
                cp_digits = self.cp_digits
                harm_digits = self.harm_digits
                restart = self._restart_requested
                self._restart_requested = False

            if restart:
                # Rewind the digit walks (carrier + counterpoint; the
                # modulator and harmonics windows follow digit_index, so
                # they realign automatically). The in-flight notes play
                # out, then _advance_note picks digit index 0.
                self.voice.digit_index = 0
                self.cp_voice.digit_index = 0

            if not digits:
                outdata.fill(0.0)
                return

            table = self._get_freq_table(p["mode"], p["base_freq"], p["subdivisions"])
            out = self.voice.render(frames, digits, mod_digits, harm_digits,
                                    self._carrier_voice_params(p), table)

            if p["cp_constant"] != "none" and cp_digits:
                cp_table = self._get_freq_table(
                    p["cp_mode"], p["cp_base_freq"] or p["base_freq"], p["subdivisions"])
                out += self.cp_voice.render(frames, cp_digits, [], [],
                                            self._cp_voice_params(p), cp_table)

            self.effects.set_amounts(chorus=p["fx_chorus"], delay=p["fx_delay"],
                                     reverb=p["fx_reverb"],
                                     reverb_room=p["fx_room_size"],
                                     reverb_damp=p["fx_damping"],
                                     reverb_width=p["fx_width"],
                                     reverb_predelay=p["fx_predelay"])
            if self.effects.active:
                out = self.effects.process(out)

            outdata[:, :] = out

            # ---- visuals tap (cheap: one ring-buffer write + a tiny dict)
            note = self.voice.note or {}
            cp_note = self.cp_voice.note if p["cp_constant"] != "none" else None
            info = {
                "digit": note.get("digit"),
                "digit_index": max(0, self.voice.digit_index - 1),
                "mod_digit": note.get("mod_digit"),
                "cp_digit": (cp_note or {}).get("digit"),
                "freqs": [round(f, 1) for f in note.get("freqs", [])],
            }
            with self.lock:
                n = len(self.visual_buffer)
                pos = self.visual_pos
                take = min(frames, n)
                first = min(take, n - pos)
                self.visual_buffer[pos:pos + first] = out[:first]
                if take > first:
                    self.visual_buffer[:take - first] = out[first:take]
                self.visual_pos = (pos + take) % n
                self.current_info = info

                # ---- recording tap: append the fresh post-FX block (no copy;
                # `out` is freshly allocated each callback and not mutated after
                # this point). Concatenation is deferred to stop_recording().
                if self._recording:
                    if self._rec_frames < self._rec_max_frames:
                        self._rec_blocks.append(out)
                        self._rec_frames += frames
                    else:
                        self._recording = False
                        self._rec_truncated = True
        except Exception:
            # Never let an exception escape into cffi. Fill silence and
            # remember it for diagnostics.
            outdata.fill(0.0)
            import traceback
            self.last_callback_error = traceback.format_exc()
