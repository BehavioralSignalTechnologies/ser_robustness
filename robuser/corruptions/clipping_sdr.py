import numpy as np

from robuser.corruptions.corruption_type import CorruptionType


def clip(x, kind, theta, asym_ratio=1.0, asym_neg=True):
    """Clip x at theta * max|x|. hard: both sides; asym: the weaker side at theta * asym_ratio; soft: theta * tanh."""
    t = theta * np.abs(x).max()
    if kind == "soft":
        return t * np.tanh(x / t)
    lo, hi = -t, t
    if kind == "asym":
        if asym_neg:
            lo = -t * asym_ratio
        else:
            hi = t * asym_ratio
    return np.clip(x, lo, hi)


def sdr_db(x, y, cap):
    """Signal-to-distortion ratio of y w.r.t. x in dB, at most cap."""
    d = np.sum((x - y) ** 2)
    return cap if d <= 0 else float(min(10 * np.log10(np.sum(x**2) / d), cap))


class ClippingSDR(CorruptionType):
    """
    Clip the audio to an exact signal-to-distortion ratio (SDR) and return it as the label.

    Unlike `clipping_distortion` (hard clipping of a random, unlogged share of samples), the clipping threshold is
    solved (bisection) so that SDR = 10 log10( sum x^2 / sum (x - clip(x))^2 ) hits the target.

    config:
        * sdr: target SDR in dB
        * clip_type: "hard" (both sides), "asym" (one side clipped more) or "soft" (tanh, no flat tops)
        * asym_ratio: for asym, the weaker side's threshold as a fraction of the other's (e.g. 0.3-0.9)
        * asym_neg: for asym, true if the negative side is the weaker one (default true)
        * sdr_cap: value of an undistorted signal (default 50)
    """

    TYPES = ("hard", "asym", "soft")

    def __init__(self, config):
        super().__init__(config)
        if "sdr" not in config:
            raise ValueError("sdr is not in the config")
        if config.get("clip_type", "hard") not in self.TYPES:
            raise ValueError(f"clip_type must be one of {self.TYPES}")
        self.kind = config.get("clip_type", "hard")
        self.cap = config.get("sdr_cap", 50.0)
        self.kw = {}
        if self.kind == "asym":
            self.kw = dict(asym_ratio=config["asym_ratio"], asym_neg=config.get("asym_neg", True))

    def solve_theta(self, x):
        """Bisection on theta (SDR increases with theta). theta > 1 is allowed: soft clipping needs it for high SDR,
        and asymmetric clipping needs up to 1 / asym_ratio before its weaker side stops clipping."""
        lo, hi = 1e-4, 50.0
        for _ in range(40):
            mid = np.sqrt(lo * hi)  # geometric bisection: theta spans orders of magnitude
            if sdr_db(x, clip(x, self.kind, mid, **self.kw), self.cap) < self.config["sdr"]:
                lo = mid
            else:
                hi = mid
        return hi

    def run_with_info(self, audio_data, sample_rate, rng=None, context=None):
        theta = self.solve_theta(audio_data)
        y = clip(audio_data, self.kind, theta, **self.kw)
        if self.kind == "soft":  # no flat tops: share of samples in the saturating region
            clipped = np.abs(audio_data) >= theta * np.abs(audio_data).max()
        else:  # hard / asym: exactly the samples the clipper changed
            clipped = np.abs(y - audio_data) > 1e-12
        return y, {"sdr_db": sdr_db(audio_data, y, self.cap), "target_sdr_db": float(self.config["sdr"]),
                   "clip_type": self.kind, "theta": float(theta), "asym_ratio": self.kw.get("asym_ratio"),
                   "clipped_pct": float(clipped.mean() * 100)}

    def run(self, audio_data, sample_rate):
        audio, _ = self.run_with_info(audio_data, sample_rate)
        return audio, None
