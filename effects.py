"""Streamable audio effects: chorus, feedback delay, Schroeder reverb.

Every effect keeps persistent state and processes (num_samples, channels)
float32 blocks, so the same EffectChain instance works both ways:

- offline (Generate path): feed the whole buffer as one big block;
- live: feed each audio-callback block in sequence — state carries across
  blocks so delay/reverb tails are continuous.

Feedback filters are vectorized by processing in chunks no longer than their
delay length: within such a chunk every feedback tap reads only already-
computed state, so each chunk is pure numpy (no per-sample Python loop).
"""

import numpy as np
from scipy import signal as sps


class _FeedbackComb:
    """y[n] = x[n] + g * y[n - delay]   (vectorized in delay-sized chunks)"""

    def __init__(self, delay_samples, gain, channels):
        self.delay = max(1, int(delay_samples))
        self.gain = float(gain)
        self.buf = np.zeros((self.delay, channels), dtype=np.float32)
        self.pos = 0  # next buffer slot to read/overwrite

    def process(self, x):
        out = np.empty_like(x)
        i = 0
        while i < len(x):
            n = min(self.delay - self.pos if self.pos else self.delay, self.delay, len(x) - i)
            sl = slice(self.pos, self.pos + n)
            y = x[i:i + n] + self.gain * self.buf[sl]
            out[i:i + n] = y
            self.buf[sl] = y
            self.pos = (self.pos + n) % self.delay
            i += n
        return out


class _LowpassComb:
    """Freeverb comb: output y[n] = buf[n-D]; the feedback path is one-pole
    lowpassed, fs[n] = (1-damp)*y[n] + damp*fs[n-1], and the buffer is
    rewritten with x[n] + feedback*fs[n].

    Mono (Freeverb runs an independent bank per channel). Vectorized in
    delay-sized chunks like _FeedbackComb: within a chunk every delayed read
    is already-written state, and the lowpass recurrence runs through
    lfilter with carried zi state — per-sample-deterministic, so block-split
    processing stays bit-identical to whole-buffer processing.
    """

    def __init__(self, delay_samples):
        self.delay = max(1, int(delay_samples))
        self.buf = np.zeros(self.delay, dtype=np.float64)
        self.pos = 0
        self.zi = np.zeros(1, dtype=np.float64)

    def process(self, x, feedback, damp):
        out = np.empty(len(x), dtype=np.float64)
        i = 0
        while i < len(x):
            n = min(self.delay - self.pos, len(x) - i)
            sl = slice(self.pos, self.pos + n)
            y = self.buf[sl].copy()
            fs, self.zi = sps.lfilter([1.0 - damp], [1.0, -damp], y, zi=self.zi)
            self.buf[sl] = x[i:i + n] + feedback * fs
            out[i:i + n] = y
            self.pos = (self.pos + n) % self.delay
            i += n
        return out


class _Allpass:
    """y[n] = -g*x[n] + x[n-delay] + g*y[n-delay]"""

    def __init__(self, delay_samples, gain, channels):
        self.delay = max(1, int(delay_samples))
        self.gain = float(gain)
        self.xbuf = np.zeros((self.delay, channels), dtype=np.float32)
        self.ybuf = np.zeros((self.delay, channels), dtype=np.float32)
        self.pos = 0

    def process(self, x):
        out = np.empty_like(x)
        i = 0
        while i < len(x):
            n = min(self.delay - self.pos if self.pos else self.delay, self.delay, len(x) - i)
            sl = slice(self.pos, self.pos + n)
            xn = x[i:i + n]
            y = -self.gain * xn + self.xbuf[sl] + self.gain * self.ybuf[sl]
            out[i:i + n] = y
            self.xbuf[sl] = xn
            self.ybuf[sl] = y
            self.pos = (self.pos + n) % self.delay
            i += n
        return out


class Chorus:
    """Sine-modulated fractional delay (~5-25 ms) mixed with the dry signal."""

    def __init__(self, sample_rate, channels, rate_hz=0.8, base_ms=15.0, depth_ms=6.0):
        self.sr = sample_rate
        self.rate = rate_hz
        self.base = base_ms * sample_rate / 1000.0
        self.depth = depth_ms * sample_rate / 1000.0
        # 1.3x depth margin: the right channel reads an extra 0.3*depth back
        self.max_delay = int(self.base + 1.3 * self.depth) + 2
        self.hist = np.zeros((self.max_delay, channels), dtype=np.float32)
        self.phase = 0.0

    def process(self, x, wet):
        n = len(x)
        hist = np.concatenate([self.hist, x])
        t = np.arange(n)
        lfo_phase = self.phase + 2.0 * np.pi * self.rate * t / self.sr
        delay = self.base + self.depth * 0.5 * (1.0 + np.sin(lfo_phase))
        read = np.arange(self.max_delay, self.max_delay + n) - delay
        wet_sig = np.empty_like(x)
        idx = np.arange(len(hist), dtype=np.float64)
        for c in range(x.shape[1]):
            # opposite-phase LFO on the right channel widens the stereo image
            r = read if c == 0 else (read - self.depth * 0.3)
            wet_sig[:, c] = np.interp(r, idx, hist[:, c])
        self.hist = hist[-self.max_delay:]
        # advance to the phase of the *next* sample so block-split processing
        # is identical to whole-buffer processing
        self.phase = float((self.phase + 2.0 * np.pi * self.rate * n / self.sr) % (2.0 * np.pi))
        return x + wet * wet_sig


class Delay:
    """Feedback echo: dry + wet * delayed-line output."""

    def __init__(self, sample_rate, channels, time_s=0.28, feedback=0.45):
        self.comb = _FeedbackComb(int(time_s * sample_rate), feedback, channels)

    def process(self, x, wet):
        echoes = self.comb.process(x) - x  # repeats only, not the dry signal
        return x + wet * echoes


class Reverb:
    """Freeverb-style reverb: per channel, 8 lowpass-damped feedback combs in
    parallel into 4 series allpasses, with pre-delay and stereo width.

    room_size / damping / width / predelay_s are live-adjustable plain floats
    (see the EffectChain thread note); heavy state is only touched by
    process(). A mid-stream pre-delay change moves the wet read offset, which
    causes a small wet-only time jump — accepted for this experimental tool.
    """

    # Classic Freeverb tunings, in samples at 44.1 kHz.
    COMB_TUNINGS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
    ALLPASS_TUNINGS = (556, 441, 341, 225)
    ALLPASS_GAIN = 0.5
    STEREO_SPREAD = 23
    MAX_PREDELAY_S = 0.25
    WET_SCALE = 0.09  # level-match: wet-path RMS ≈ 0.36x dry at room=0.5, wet=1

    def __init__(self, sample_rate, channels):
        self.sr = sample_rate
        self.channels = channels
        scale = sample_rate / 44100.0
        spread = int(self.STEREO_SPREAD * scale)
        self.combs = [
            [_LowpassComb(int(d * scale) + spread * c) for d in self.COMB_TUNINGS]
            for c in range(channels)
        ]
        self.allpasses = [
            [_Allpass(int(d * scale) + spread * c, self.ALLPASS_GAIN, 1)
             for d in self.ALLPASS_TUNINGS]
            for c in range(channels)
        ]
        self.pre_hist = np.zeros((int(self.MAX_PREDELAY_S * sample_rate) + 1, channels),
                                 dtype=np.float32)
        self.room_size = 0.5
        self.damping = 0.5
        self.width = 1.0
        self.predelay_s = 0.0

    def process(self, x, wet):
        d = int(np.clip(self.predelay_s, 0.0, self.MAX_PREDELAY_S) * self.sr)
        hist = np.concatenate([self.pre_hist, x])
        xin = hist[len(hist) - len(x) - d:len(hist) - d]
        self.pre_hist = hist[-len(self.pre_hist):]

        feedback = 0.7 + 0.28 * min(float(self.room_size), 0.98)
        damp = 0.4 * float(np.clip(self.damping, 0.0, 1.0))

        wet_sig = np.empty_like(x, dtype=np.float64)
        for c in range(self.channels):
            ch = self.WET_SCALE * sum(
                comb.process(xin[:, c], feedback, damp) for comb in self.combs[c])
            ch = ch[:, None]
            for ap in self.allpasses[c]:
                ch = ap.process(ch)
            wet_sig[:, c] = ch[:, 0]

        if self.channels == 2:
            w1 = float(self.width) / 2.0 + 0.5
            w2 = (1.0 - float(self.width)) / 2.0
            left = wet_sig[:, 0] * w1 + wet_sig[:, 1] * w2
            right = wet_sig[:, 1] * w1 + wet_sig[:, 0] * w2
            wet_sig = np.stack([left, right], axis=1)

        return x + wet * wet_sig.astype(np.float32)


class EffectChain:
    """Chorus → delay → reverb with 0-1 wet amounts; 0 bypasses an effect.

    Thread note for live use: set_amounts() only writes floats, and process()
    only reads them, so a torn read is harmless; the heavy state lives inside
    the effect objects, which only process() touches.
    """

    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chorus_amt = 0.0
        self.delay_amt = 0.0
        self.reverb_amt = 0.0
        self.chorus = Chorus(sample_rate, channels)
        self.delay = Delay(sample_rate, channels)
        self.reverb = Reverb(sample_rate, channels)

    def set_amounts(self, chorus=None, delay=None, reverb=None, reverb_room=None,
                    reverb_damp=None, reverb_width=None, reverb_predelay=None):
        if chorus is not None:
            self.chorus_amt = float(chorus)
        if delay is not None:
            self.delay_amt = float(delay)
        if reverb is not None:
            self.reverb_amt = float(reverb)
        if reverb_room is not None:
            self.reverb.room_size = float(reverb_room)
        if reverb_damp is not None:
            self.reverb.damping = float(reverb_damp)
        if reverb_width is not None:
            self.reverb.width = float(reverb_width)
        if reverb_predelay is not None:
            self.reverb.predelay_s = float(reverb_predelay)

    @property
    def active(self):
        return self.chorus_amt > 0 or self.delay_amt > 0 or self.reverb_amt > 0

    def process(self, x):
        """x: float32 array (num_samples, channels); returns same shape."""
        if self.chorus_amt > 0:
            x = self.chorus.process(x, self.chorus_amt)
        if self.delay_amt > 0:
            x = self.delay.process(x, self.delay_amt)
        if self.reverb_amt > 0:
            x = self.reverb.process(x, self.reverb_amt)
        return x.astype(np.float32, copy=False)

    def reset(self):
        """Clear all tails (e.g. when live playback restarts)."""
        self.__init__(self.sample_rate, self.channels)


def apply_effects_offline(audio, sample_rate, chorus=0.0, delay=0.0, reverb=0.0,
                          tail_seconds=1.5, reverb_room=0.5, reverb_damp=0.5,
                          reverb_width=1.0, reverb_predelay=0.0):
    """
    One-shot convenience for the Generate path. Appends silence so delay and
    reverb tails ring out (bigger rooms get a longer tail), then runs the
    buffer through a fresh chain.
    """
    if not (chorus > 0 or delay > 0 or reverb > 0):
        return audio
    if audio.ndim == 1:
        audio = audio[:, None]
    if reverb > 0:
        tail_seconds = max(tail_seconds, 1.5 + 6.0 * float(reverb_room))
    tail = np.zeros((int(tail_seconds * sample_rate), audio.shape[1]), dtype=np.float32)
    padded = np.concatenate([audio, tail])
    chain = EffectChain(sample_rate, audio.shape[1])
    chain.set_amounts(chorus=chorus, delay=delay, reverb=reverb,
                      reverb_room=reverb_room, reverb_damp=reverb_damp,
                      reverb_width=reverb_width, reverb_predelay=reverb_predelay)
    return chain.process(padded)
