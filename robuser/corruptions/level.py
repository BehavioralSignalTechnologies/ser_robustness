import numpy as np

from robuser.corruptions.corruption_type import CorruptionType


class Level(CorruptionType):
    """
    Set the output level (a random volume). Scales the whole signal, so labels measured earlier in a chain (SNR, SDR,
    C50) are unchanged. Put it before compression: that step goes through 16-bit audio, which clips anything above 1.

    config:
        * mode: "rms" (a random RMS level, default) or "peak" (the peak normalised near full scale)
        * rms_db: [min, max] RMS level in dBFS for mode "rms" (default [-35, -15])
        * peak: [min, max] peak for mode "peak" (default [0.90, 0.99])
        * max_peak: the peak never exceeds this, so the level never clips (default 0.99). In mode "rms" a file with a
          high peak-to-RMS ratio therefore ends up quieter than the drawn target; the measured level is returned.
        * seed: seed of the generator used when run_with_info gets no rng (default 42)
    """

    MODES = ("rms", "peak")

    def __init__(self, config):
        super().__init__(config)
        self.mode = config.get("mode", "rms")
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.rms_db = tuple(config.get("rms_db", (-35.0, -15.0)))
        self.peak = tuple(config.get("peak", (0.90, 0.99)))
        self.max_peak = float(config.get("max_peak", 0.99))
        if len(self.rms_db) != 2 or self.rms_db[0] > self.rms_db[1] or self.rms_db[1] > 0:
            raise ValueError("rms_db must be [min, max] dBFS with min <= max <= 0")
        if len(self.peak) != 2 or not 0 < self.peak[0] <= self.peak[1] <= 1:
            raise ValueError("peak must be [min, max] with 0 < min <= max <= 1")
        if not 0 < self.max_peak <= 1:
            raise ValueError("max_peak must be in (0, 1]")
        self.rng = np.random.default_rng(config.get("seed", 42))

    def run_with_info(self, audio_data, sample_rate, rng=None):
        """
        Scale the signal and return gain_db, mode, the drawn target (target_rms_dbfs or target_peak) and the measured
        output level (rms_dbfs, peak).
        """
        rng = rng if rng is not None else self.rng
        peak = np.abs(audio_data).max()
        info = {"mode": self.mode}
        if self.mode == "peak":
            target = float(rng.uniform(*self.peak))
            gain = target / (peak + 1e-12)
            info["target_peak"] = target
        else:
            target = float(rng.uniform(*self.rms_db))
            gain = 10 ** (target / 20) / (np.sqrt(np.mean(audio_data ** 2)) + 1e-12)
            gain = min(gain, self.max_peak / (peak + 1e-12))
            info["target_rms_dbfs"] = target
        out = (audio_data * gain).astype(audio_data.dtype)
        info.update(gain_db=float(20 * np.log10(gain)),
                    rms_dbfs=float(10 * np.log10(np.mean(out.astype(np.float64) ** 2) + 1e-20)),
                    peak=float(np.abs(out).max()))
        return out, info

    def run(self, audio_data, sample_rate):
        audio, _ = self.run_with_info(audio_data, sample_rate)
        return audio, None
