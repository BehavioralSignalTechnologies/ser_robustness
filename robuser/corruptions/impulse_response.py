import os
import random

import librosa
import numpy as np
import pyroomacoustics as pra
from audiomentations import ApplyImpulseResponse
from scipy.signal import convolve

from robuser.corruptions.corruption_type import CorruptionType
from robuser.corruptions.utils import audio_files_in, load_channel

# C50 / DRR (calculate_c50): the direct sound is the window from DIRECT_WINDOW_S before to DIRECT_WINDOW_S after the
# direct-path peak (+-8 ms reproduces the ACE corpus DRR values); early sound is up to C50_BOUNDARY_S after the peak.
DIRECT_WINDOW_S = 0.008
C50_BOUNDARY_S = 0.050


def load_impulse_response(impulse_response_path, channel=None, sample_rate=None):
    """
    Load an impulse response
        :param channel: None to downmix a multichannel file (default), or the channel index
        :param sample_rate: the rate to resample to (default: the file's own rate)
        :return: (impulse response, sample rate)
    """
    if sample_rate is None:
        sample_rate = librosa.get_samplerate(impulse_response_path)
    impulse_response, _ = load_channel(impulse_response_path, sample_rate, channel)
    return impulse_response, sample_rate


def calculate_rt60(impulse_response_path, channel=None, sample_rate=None):
    """
    Calculate the RT60 of the impulse response
        :param impulse_response_path: the path to the impulse response
        :param channel: None to downmix a multichannel file (default), or the channel index
        :param sample_rate: the rate to compute it at (default: the file's own rate)
        :return: the RT60 in seconds
    """
    impulse_response, sample_rate = load_impulse_response(impulse_response_path, channel, sample_rate)
    # Normalize the impulse response
    norm_impulse_response = impulse_response / max(abs(impulse_response))
    rt60 = pra.experimental.measure_rt60(norm_impulse_response, fs=sample_rate)
    return rt60


def direct_path_start(impulse_response, sample_rate):
    """Index where the direct sound starts: DIRECT_WINDOW_S before the direct-path peak (argmax |h|)."""
    peak = int(np.argmax(np.abs(impulse_response)))
    return max(0, peak - int(round(DIRECT_WINDOW_S * sample_rate))), peak


def calculate_c50(impulse_response_path, channel=None, sample_rate=None):
    """
    Calculate the C50 clarity and the direct-to-reverberant ratio (DRR) of the impulse response
        C50 = 10 log10( E[t0 - 8 ms, t0 + 50 ms) / E[t0 + 50 ms, end) )
        DRR = 10 log10( E[t0 - 8 ms, t0 + 8 ms)  / E[t0 + 8 ms, end) ),     t0 = direct-path peak
    Unlike RT60 (a property of the room), C50 also depends on the source-microphone distance.
        :param impulse_response_path: the path to the impulse response
        :param channel: None to downmix a multichannel file (default), or the channel index
        :param sample_rate: the rate to compute it at (default: the file's own rate). Compute it at the rate of the
            audio it is applied to: the values depend on the bandwidth.
        :return: (C50 in dB, DRR in dB)
    """
    impulse_response, sample_rate = load_impulse_response(impulse_response_path, channel, sample_rate)
    energy = impulse_response.astype(np.float64) ** 2
    start, peak = direct_path_start(impulse_response, sample_rate)
    early_end = peak + int(round(C50_BOUNDARY_S * sample_rate))
    direct_end = peak + int(round(DIRECT_WINDOW_S * sample_rate))

    def ratio_db(num, den):
        return float(10 * np.log10(num / den)) if num > 0 and den > 0 else float("nan")

    c50 = ratio_db(energy[start:early_end].sum(), energy[early_end:].sum())
    drr = ratio_db(energy[start:direct_end].sum(), energy[direct_end:].sum())
    return c50, drr


class AddImpulseResponse(CorruptionType):
    """
    Convolve the audio with a randomly selected impulse response.

    config:
        `ir_path`  (str/Path): A path or list of paths to audio file(s) and/or folder(s) with audio files.
        `rt60_range` (float, float): The range of the RT60 in seconds of the impulse responses to be used.
            Required when ir_path contains a folder; optional (no filtering) for files.
    optional (defaults keep the original behaviour):
        `channel` (int): the channel of a multichannel impulse response to apply. Omitted: downmixed (the sum of
            responses at different microphones is not a physical response, so give it for microphone arrays).
        `align` (bool): cut the impulse response where its direct sound starts (8 ms before the direct-path peak,
            the start calculate_c50 measures from), so the output is not delayed by the response's lead-in and
            the label describes exactly the applied response. Default false.

    *download the echo thief impulse response dataset: http://www.echothief.com/downloads/
    """

    def __init__(self, config):
        """
        Initialize the ImpulseResponse class

            :param config: dictionary with the configuration parameters
        """
        super().__init__(config)

        if "ir_path" not in config:
            raise ValueError("The ir_path is not provided. " +
                             "Please provide the path to the impulse response files.")

        self.ir_path = config["ir_path"]
        files, any_folder = audio_files_in(self.ir_path, sort=False)  # os.walk order, as before

        if "rt60_range" in config:
            if not isinstance(config["rt60_range"], (list, tuple)) or len(config["rt60_range"]) != 2:
                raise ValueError("rt60_range must be a list or tuple of [min_rt60, max_rt60]")
            self.rt60_min, self.rt60_max = config["rt60_range"]
        elif any_folder:
            raise ValueError("The rt60_range is not provided. " +
                             "Please provide the range of RT60 in seconds.")

        channel = config.get("channel")
        if channel is not None and not (isinstance(channel, int) and channel >= 0):
            raise ValueError("channel must be omitted or a non-negative integer")
        self.channel = channel
        self.align = bool(config.get("align", False))

        random.seed(42)
        if "rt60_range" in config:
            self.selected_irs = [f for f in files if self.rt60_min <= calculate_rt60(f) <= self.rt60_max]
        else:
            self.selected_irs = files
        print(f"Selected {len(self.selected_irs)} impulse responses from {self.ir_path}"
              + (f" with RT60 in range [{self.rt60_min}, {self.rt60_max}]" if "rt60_range" in config else ""))
        if not self.selected_irs:
            raise ValueError(f"No impulse responses selected from {self.ir_path}")

    def select_ir(self, rng=None):
        if rng is None:
            return random.choice(self.selected_irs)
        return self.selected_irs[int(rng.integers(len(self.selected_irs)))]

    def convolve(self, audio_data, sample_rate, ir_wav_path):
        """Convolve with one impulse response. Without channel / align, audiomentations' ApplyImpulseResponse (the
        original behaviour); otherwise the same operation on the chosen channel: convolve, scale to peak 0.5, cut to
        the input length."""
        if self.channel is None and not self.align:
            transform = ApplyImpulseResponse(
                ir_path=ir_wav_path,
                p=1.0
            )
            return transform(audio_data, sample_rate)
        ir, _ = load_impulse_response(ir_wav_path, self.channel, sample_rate)
        if self.align:
            ir = ir[direct_path_start(ir, sample_rate)[0]:]
        out = convolve(audio_data, ir.astype(audio_data.dtype))
        max_value = np.abs(out).max()
        if max_value > 0.0:
            out = out * (0.5 / max_value)
        return out[: len(audio_data)].astype(audio_data.dtype)

    def run_with_info(self, audio_data, sample_rate, rng=None):
        """
        Convolve and return what was applied: ir_path, channel, and the response's rt60_s, c50_db and drr_db,
        computed on the applied channel at the audio's sample rate.

        With rng=None the response is chosen with Python's random (the original behaviour), otherwise from rng.
        """
        ir_wav_path = self.select_ir(rng)
        out = self.convolve(audio_data, sample_rate, ir_wav_path)
        c50, drr = calculate_c50(ir_wav_path, self.channel, sample_rate)
        return out, {"ir_path": ir_wav_path, "channel": self.channel,
                     "rt60_s": float(calculate_rt60(ir_wav_path, self.channel, sample_rate)),
                     "c50_db": c50, "drr_db": drr}

    def run(self, audio_data, sample_rate):
        """
        Run the impulse response method

            :param audio_data: numpy array with the audio data
            :param sample_rate: the sample rate

            :return: the augmented audio data (numpy array) and the applied impulse response
        """
        ir_wav_path = self.select_ir()
        return self.convolve(audio_data, sample_rate, ir_wav_path), ir_wav_path
