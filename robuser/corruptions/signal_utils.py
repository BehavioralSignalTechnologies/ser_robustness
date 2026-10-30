"""Signal helpers for the labelled corruptions (noise_snr, rir, clipping_sdr, level, telephony_codec, chain).

The labelled corruptions measure the degradation they apply (SNR, C50, SDR) from the separate components, so the
level is exact. Speech activity is taken from the clean input (energy VAD) and the SNR is measured over those frames.
"""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

VAD_FRAME_S = 0.02
VAD_DB_BELOW_PEAK = 35.0


def speech_mask(signal, sample_rate):
    """Sample-level speech mask from a clean signal: 20 ms frames within 35 dB of the 95th-percentile frame energy."""
    frame = int(round(VAD_FRAME_S * sample_rate))
    n = len(signal) // frame
    energy = 10 * np.log10(np.mean(signal[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    active = energy > (np.percentile(energy, 95) - VAD_DB_BELOW_PEAK)
    mask = np.zeros(len(signal), bool)
    mask[: n * frame] = np.repeat(active, frame)
    return mask


def power(signal, mask):
    return float(np.mean(signal[mask] ** 2))


def resample(signal, sr_in, sr_out):
    if sr_in == sr_out:
        return signal
    g = gcd(sr_in, sr_out)
    return resample_poly(signal, sr_out // g, sr_in // g)


def read_segment(path, n_out, sample_rate, rng, channel=None):
    """Read a random window of n_out samples (at sample_rate) from an audio file: mono, resampled, looped if short.

    Returns (signal, offset_s, channel). The window start and, if channel is None, the channel are drawn from rng.
    """
    info = sf.info(path)
    n_src = int(np.ceil(n_out * info.samplerate / sample_rate)) + 64
    if info.frames > n_src:
        start = int(rng.integers(0, info.frames - n_src))
        x, sr = sf.read(path, start=start, frames=n_src, dtype="float32", always_2d=True)
        offset = start / info.samplerate
    else:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        offset = 0.0
    ch = int(rng.integers(0, x.shape[1])) if channel is None else channel
    x = x[:, ch]
    if sr != sample_rate:
        x = resample(x, sr, sample_rate).astype(np.float32)
    if len(x) < n_out:  # loop short clips: the noise always covers the whole signal
        x = np.tile(x, int(np.ceil(n_out / max(len(x), 1))))
    return x[:n_out], offset, ch
