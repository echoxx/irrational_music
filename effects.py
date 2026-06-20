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
    """Classic Schroeder reverb: 4 parallel combs into 2 series allpasses."""

    COMB_MS = (29.7, 37.1, 41.1, 43.7)
    COMB_GAINS = (0.805, 0.827, 0.783, 0.764)
    ALLPASS_MS = (5.0, 1.7)
    ALLPASS_GAIN = 0.7

    def __init__(self, sample_rate, channels):
        ms = sample_rate / 1000.0
        # slight per-channel detuning of comb delays decorrelates L/R
        self.combs = [
            _FeedbackComb(int(d * ms) + (1 if i % 2 else 0), g, channels)
            for i, (d, g) in enumerate(zip(self.COMB_MS, self.COMB_GAINS))
        ]
        self.allpasses = [_Allpass(int(d * ms), self.ALLPASS_GAIN, channels)
                          for d in self.ALLPASS_MS]

    def process(self, x, wet):
        wet_sig = sum(c.process(x) for c in self.combs) / len(self.combs)
        for ap in self.allpasses:
            wet_sig = ap.process(wet_sig)
        return x + wet * wet_sig


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

    def set_amounts(self, chorus=None, delay=None, reverb=None):
        if chorus is not None:
            self.chorus_amt = float(chorus)
        if delay is not None:
            self.delay_amt = float(delay)
        if reverb is not None:
            self.reverb_amt = float(reverb)

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
                          tail_seconds=1.5):
    """
    One-shot convenience for the Generate path. Appends silence so delay and
    reverb tails ring out, then runs the buffer through a fresh chain.
    """
    if not (chorus > 0 or delay > 0 or reverb > 0):
        return audio
    if audio.ndim == 1:
        audio = audio[:, None]
    tail = np.zeros((int(tail_seconds * sample_rate), audio.shape[1]), dtype=np.float32)
    padded = np.concatenate([audio, tail])
    chain = EffectChain(sample_rate, audio.shape[1])
    chain.set_amounts(chorus=chorus, delay=delay, reverb=reverb)
    return chain.process(padded)
