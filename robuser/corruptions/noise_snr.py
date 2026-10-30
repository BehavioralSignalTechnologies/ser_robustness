import numpy as np

from robuser.corruptions.corruption_type import CorruptionType
from robuser.corruptions.signal_utils import power, read_segment, speech_mask


class NoiseSNR(CorruptionType):
    """
    Add noise at an exact speech-to-noise ratio, measured over the speech frames of the clean input.

    Unlike `content` (SNR over the whole file, short noise zero-padded, output re-normalised), the noise always covers
    the whole signal (a random window, looped if the file is short), the output level is left to a later `level` step,
    and the measured SNR is returned as the label.

    config:
        * snr: target SNR in dB
        * noise_path: noise file (a random window is used)  -- or --
        * speech_paths: list of speech files summed at equal power (1 = competing talker, 3-7 = babble)
        * snr_cap: optional upper bound of the returned label (e.g. 50: noise silent during speech counts as clean)
        * min_noise_power: if the noise power over the speech frames is <= this, the whole-window power is used
          instead (default 1e-10)
    """

    def __init__(self, config):
        super().__init__(config)
        if "snr" not in config:
            raise ValueError("snr is not in the config")
        if ("noise_path" in config) == ("speech_paths" in config):
            raise ValueError("exactly one of noise_path or speech_paths is required")

    def _noise(self, n_out, sample_rate, rng):
        if "speech_paths" in self.config:
            noise = np.zeros(n_out, np.float32)
            for path in self.config["speech_paths"]:  # equal-power talkers, random windows
                x, _, _ = read_segment(path, n_out, sample_rate, rng, channel=0)
                noise += x / np.sqrt(np.mean(x**2) + 1e-12)
            info = {"noise_path": "|".join(self.config["speech_paths"]), "noise_offset_s": None, "noise_channel": 0,
                    "n_talkers": len(self.config["speech_paths"])}
        else:
            noise, offset, channel = read_segment(self.config["noise_path"], n_out, sample_rate, rng)
            info = {"noise_path": self.config["noise_path"], "noise_offset_s": offset, "noise_channel": channel,
                    "n_talkers": 0}
        return noise, info

    def run_with_info(self, audio_data, sample_rate, rng=None, context=None):
        rng = rng if rng is not None else np.random.default_rng()
        mask = (context or {}).get("speech_mask")
        if mask is None:
            mask = speech_mask(audio_data, sample_rate)
        noise, info = self._noise(len(audio_data), sample_rate, rng)
        noise = noise.astype(audio_data.dtype, copy=False)

        p_signal = power(audio_data, mask)
        p_noise = power(noise, mask)
        if p_noise <= self.config.get("min_noise_power", 1e-10):
            p_noise = float(np.mean(noise**2)) + 1e-12
        noise = noise * np.sqrt(p_signal / (p_noise * 10 ** (self.config["snr"] / 10)))
        snr = 10 * np.log10(p_signal / (power(noise, mask) + 1e-20))
        if self.config.get("snr_cap") is not None:
            snr = min(snr, self.config["snr_cap"])
        info.update(snr_db=float(snr), target_snr_db=float(self.config["snr"]))
        return audio_data + noise, info

    def run(self, audio_data, sample_rate):
        audio, info = self.run_with_info(audio_data, sample_rate)
        return audio, info["noise_path"]
