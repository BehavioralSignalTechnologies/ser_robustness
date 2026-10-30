import numpy as np

from robuser.corruptions.corruption_type import CorruptionType


class Level(CorruptionType):
    """
    Set the output level. Scales the whole signal, so labels measured earlier in a chain (SNR, SDR) are unchanged.

    config:
        * mode: "rms" (random RMS level, default) or "peak" (peak normalised near full scale)
        * rms_db: [min, max] RMS level in dBFS for mode "rms" (default [-35, -15])
        * peak: [min, max] peak for mode "peak" (default [0.90, 0.99])
        * max_peak: the peak never exceeds this, so the level never clips (default 0.99)
    The level is drawn from the rng passed to run_with_info.
    """

    def run_with_info(self, audio_data, sample_rate, rng=None, context=None):
        rng = rng if rng is not None else np.random.default_rng()
        max_peak = self.config.get("max_peak", 0.99)
        peak = np.abs(audio_data).max()
        if self.config.get("mode", "rms") == "peak":
            gain = rng.uniform(*self.config.get("peak", (0.90, 0.99))) / (peak + 1e-12)
        else:
            rms_db = rng.uniform(*self.config.get("rms_db", (-35.0, -15.0)))
            gain = 10 ** (rms_db / 20) / (np.sqrt(np.mean(audio_data**2)) + 1e-12)
            gain = min(gain, max_peak / (peak + 1e-12))
        return (audio_data * gain).astype(np.float32), {"gain_db": float(20 * np.log10(gain)),
                                                        "full_scale": self.config.get("mode", "rms") == "peak"}

    def run(self, audio_data, sample_rate):
        audio, _ = self.run_with_info(audio_data, sample_rate)
        return audio, None
