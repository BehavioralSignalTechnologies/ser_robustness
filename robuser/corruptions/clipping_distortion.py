import numpy as np
from audiomentations import ClippingDistortion

from robuser.corruptions.corruption_type import CorruptionType
from robuser.corruptions.utils import sdr_db


class AddClippingDistortion(CorruptionType):
    """
    Distort signal by clipping a random percentage of points

        :param min_percentile_threshold: lower bound on the total percent of samples that will be clipped
        :param max_percentile_threshold: upper bound on the total percent of samples that will be clipped

    The percentage of points that will be clipped is drawn from a uniform distribution between
    the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
    30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.

    With `sdr` (target signal-to-distortion ratio in dB) the clipped share is not drawn: it is solved (bisection, the
    SDR decreases as the clipped share grows) so that SDR = 10 log10( sum x^2 / sum (x - clipped x)^2 ) hits the
    target, and max_percentile_threshold is not needed. Options in this mode:
        :param clip_type: "hard" (default; the same clipping as above), "asym" (the clipped share is split unevenly
            between the two sides) or "soft" (tanh saturation at a threshold solved for the target SDR)
        :param asym_ratio: for asym, the less clipped side's share of samples as a fraction of the other side's
            (0 < asym_ratio <= 1; 1 = hard)
        :param asym_neg: for asym, true (default) if the negative side is the more clipped one
    """

    CLIP_TYPES = ("hard", "asym", "soft")
    BISECTION_STEPS = 50

    def __init__(self, config):
        """
        Initialize the ClippingDistortion class

            :param config: dictionary with the configuration parameters
        """
        super().__init__(config)

        self.target_sdr = config.get("sdr")
        if self.target_sdr is None:
            if "max_percentile_threshold" not in config:
                raise ValueError("'max_percentile_threshold' is a required parameter")

            self.max_percentile_threshold = config["max_percentile_threshold"]

            if not isinstance(self.max_percentile_threshold, int):
                raise ValueError("'max_percentile_threshold' must be an integer")
        else:
            self.target_sdr = float(self.target_sdr)

        self.clip_type = config.get("clip_type", "hard")
        if self.clip_type not in self.CLIP_TYPES:
            raise ValueError(f"clip_type must be one of {self.CLIP_TYPES}")
        if self.clip_type != "hard" and self.target_sdr is None:
            raise ValueError("clip_type 'asym' / 'soft' needs a target 'sdr'")
        self.asym_ratio = float(config.get("asym_ratio", 1.0))
        self.asym_neg = bool(config.get("asym_neg", True))
        if self.clip_type == "asym" and not 0 < self.asym_ratio <= 1:
            raise ValueError("asym_ratio must be in (0, 1]")

        self.p_clipping = 1.0

    def percentile_thresholds(self, samples, percentile):
        """(lower, upper) clipping thresholds for `percentile` % of clipped samples: the lower / upper percentiles, as
        audiomentations uses, with the share split between the sides by asym_ratio for asym."""
        ratio = self.asym_ratio if self.clip_type == "asym" else 1.0
        more, less = percentile / (1 + ratio), percentile * ratio / (1 + ratio)
        low_share, high_share = (more, less) if self.asym_neg else (less, more)
        return np.percentile(samples, [low_share, 100 - high_share])

    def clip_percentile(self, samples, percentile):
        """Clip `percentile` % of the samples (see percentile_thresholds)."""
        lower, upper = self.percentile_thresholds(samples, percentile)
        return np.clip(samples, lower, upper).astype(samples.dtype)

    def clip_soft(self, samples, theta):
        """tanh saturation at theta x the peak: no flat tops."""
        threshold = theta * np.abs(samples).max()
        return (threshold * np.tanh(samples / threshold)).astype(samples.dtype)

    def solve(self, samples):
        """Bisection for the target SDR. hard / asym: over the clipped share (percent of samples); soft: over theta
        (geometric, as it spans orders of magnitude). Returns (clipped samples, solved parameter)."""
        if self.clip_type == "soft":
            lo, hi, apply = 1e-4, 50.0, self.clip_soft
        else:
            lo, hi, apply = 0.0, 100.0, self.clip_percentile
        for _ in range(self.BISECTION_STEPS):
            mid = np.sqrt(lo * hi) if self.clip_type == "soft" else (lo + hi) / 2
            too_distorted = sdr_db(samples, apply(samples, mid)) < self.target_sdr
            if self.clip_type == "soft":  # SDR grows with theta
                lo, hi = (mid, hi) if too_distorted else (lo, mid)
            else:  # SDR falls as the clipped share grows
                lo, hi = (lo, mid) if too_distorted else (mid, hi)
        param = hi if self.clip_type == "soft" else lo
        return apply(samples, param), param

    def run_with_info(self, audio_data, sample_rate, rng=None):
        """
        Clip and return what was applied: sdr_db (measured), clip_type, clipped_pct, and with `sdr` also
        target_sdr_db and, for hard / asym, the solved percentile and the lower / upper thresholds as fractions of the
        peak, or for soft the solved theta. Without `sdr` the clipped share is drawn as before (Python's random,
        through audiomentations); with `sdr` nothing is random.
        """
        if self.target_sdr is None:
            transform = ClippingDistortion(
                max_percentile_threshold=self.max_percentile_threshold,
                p=self.p_clipping
            )
            out = transform(audio_data, sample_rate)
            return out, {"sdr_db": sdr_db(audio_data, out), "clip_type": "hard",
                         "clipped_pct": float(transform.parameters["percentile_threshold"])}
        out, param = self.solve(audio_data)
        info = {"sdr_db": sdr_db(audio_data, out), "target_sdr_db": self.target_sdr, "clip_type": self.clip_type}
        if self.clip_type == "soft":  # no flat tops: share of samples in the saturating region
            info.update(theta=float(param),
                        clipped_pct=float(np.mean(np.abs(audio_data) >= param * np.abs(audio_data).max()) * 100))
        else:
            lower, upper = self.percentile_thresholds(audio_data, param)
            peak = np.abs(audio_data).max()
            info.update(percentile=float(param), clipped_pct=float(np.mean(out != audio_data) * 100),
                        lower_threshold=float(lower / peak), upper_threshold=float(upper / peak))
        if self.clip_type == "asym":
            info.update(asym_ratio=self.asym_ratio, asym_neg=self.asym_neg)
        return out, info

    def run(self, audio_data, sample_rate):
        """
        Run the clipping distortion augmentation method

            :param audio_data: numpy array with the audio data
            :param sample_rate: the sample rate

            :return: the augmented audio data (numpy array)
        """
        if self.target_sdr is not None:
            return self.run_with_info(audio_data, sample_rate)[0], None
        transform = ClippingDistortion(
            max_percentile_threshold=self.max_percentile_threshold,
            p=self.p_clipping
        )
        return transform(audio_data, sample_rate), None
