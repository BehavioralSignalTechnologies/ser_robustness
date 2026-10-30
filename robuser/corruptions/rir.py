import numpy as np
from scipy.signal import fftconvolve

from robuser.corruptions.corruption_type import CorruptionType
from robuser.rir_labels import labels_for, load_rir

PRE_PEAK_S = 0.002  # kept before the direct-path peak (band-limited ringing)


class RIRConvolution(CorruptionType):
    """
    Convolve the audio with one room impulse response and return its acoustic labels.

    Unlike `impulse_response` (random RIR within an RT60 range, RT60 only), the RIR is given per file, the output is
    aligned to the direct path (no time shift, same length as the input) and C50 / DRR / T60 are returned, computed at
    the audio's sample rate (see robuser.rir_labels).

    config:
        * rir_path: RIR file
        * channel: RIR channel (default 0)
        * c50_cap: optional upper bound of the returned c50_db (c50_raw_db keeps the uncapped value)
    """

    def __init__(self, config):
        super().__init__(config)
        if "rir_path" not in config:
            raise ValueError("rir_path is not in the config")

    def run_with_info(self, audio_data, sample_rate, rng=None, context=None):
        channel = int(self.config.get("channel", 0))
        h = load_rir(self.config["rir_path"], sample_rate, channel)
        labels = labels_for(h, sample_rate)

        t0 = int(np.argmax(np.abs(h)))
        h = h[max(0, t0 - int(PRE_PEAK_S * sample_rate)) :]  # direct path at the start
        h = h / np.abs(h).max()
        pre = int(np.argmax(np.abs(h)))
        out = fftconvolve(audio_data, h)[pre : pre + len(audio_data)].astype(audio_data.dtype)

        c50 = labels["c50_db"]
        if self.config.get("c50_cap") is not None:
            c50 = min(c50, self.config["c50_cap"])
        return out, {"rir_path": self.config["rir_path"], "rir_channel": channel, "c50_db": float(c50),
                     "c50_raw_db": labels["c50_db"], "drr_db": labels["drr_db"], "t60_s": labels["t60_s"]}

    def run(self, audio_data, sample_rate):
        audio, info = self.run_with_info(audio_data, sample_rate)
        return audio, info["rir_path"]
