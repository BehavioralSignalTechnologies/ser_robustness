# ESC50 dataset: https://github.com/karolpiczak/ESC-50
# UrbanSound8K dataset: Use the script here: https://github.com/soundata/soundata#quick-example
# MUSAN dataset: https://www.openslr.org/resources/17/musan.tar.gz

import os
import random
import warnings
from itertools import cycle

import librosa
import numpy as np

from robuser.corruptions.corruption_type import CorruptionType
from robuser.corruptions.utils import audio_files_in, normalize_audio, speech_mask


class ContentCorruption(CorruptionType):
    """
    Class that augments the audio data with content from a sound dataset
    config should contain:
        * content_dataset_path: a folder, an audio file, or a list of files / folders
        * snr: the signal-to-noise ratio in dB
    optional (defaults keep the original behaviour):
        * pad: "zero" (default; a short noise is zero-padded) or "loop" (a short noise is repeated to cover the file)
        * snr_mode: "whole" (default; SNR over the whole file) or "speech" (SNR over the speech frames of the input,
          so the label does not depend on how much silence the file has)
        * mix: "pick" (default; one file) or "sum" (every file, each normalised to equal power, summed: babble)
        * channel: omitted (default; multichannel noise is downmixed), a channel index, or "random"
    """

    PAD = ("zero", "loop")
    SNR_MODES = ("whole", "speech")
    MIXES = ("pick", "sum")
    # snr_mode "speech": the noise counts as silent during speech if its power there is this far below its own
    # whole-window power (the gain would blow up, or audible noise in the pauses would be labelled clean)
    SILENT_NOISE_DB = 30.0

    def __init__(self, config):
        """
        Initialize the ContentAugmentation class

        :param config: dictionary with the configuration parameters
        """
        super().__init__(config)
        if "content_dataset_path" not in config:
            raise ValueError("content_dataset_path is not in the config")
        if "snr" not in config:
            raise ValueError("SNR is not in the config")
        for key, allowed in (("pad", self.PAD), ("snr_mode", self.SNR_MODES), ("mix", self.MIXES)):
            if config.get(key, allowed[0]) not in allowed:
                raise ValueError(f"{key} must be one of {allowed}")
        channel = config.get("channel")
        if channel is not None and channel != "random" and not (isinstance(channel, int) and channel >= 0):
            raise ValueError("channel must be omitted, a non-negative integer or 'random'")
        self.pad = config.get("pad", "zero")
        self.snr_mode = config.get("snr_mode", "whole")
        self.mix = config.get("mix", "pick")
        self.channel = channel

        self.dataset_path = config["content_dataset_path"]
        self.audio_files = self.get_audio_files()
        random.seed(42)
        self.random_audio_files = random.choices(self.audio_files, k=len(self.audio_files) * 100)
        self.random_audio_generator = cycle(self.random_audio_files)

    def get_audio_files(self):
        """
        Get the audio files from the dataset
        Returns:
            a sorted list with the audio files
        """
        audio_files, any_folder = audio_files_in(self.dataset_path)
        if not audio_files:
            raise ValueError(f"No audio files in {self.dataset_path}")
        # basenames identify the noise in run()'s output, so they must be unique when a folder is walked
        if any_folder and len(audio_files) != len(set([os.path.basename(file) for file in audio_files])):
            raise ValueError("There are duplicate filenames in the dataset")
        return audio_files

    def calculate_snr(self, signal, noise, mask=None):
        """Calculates the snr from signal and noise

        Args:
            signal (np.array): clean signal
            noise (np.array): added noise
            mask (np.array): optional boolean mask of the samples to measure over (default: all)

        Returns:
            float: snr
        """
        if mask is not None:
            signal, noise = signal[mask], noise[mask]
        # Calculate the power of the signal
        signal_power = np.sum(signal ** 2) / len(signal)

        # Calculate the power of the noise
        noise_power = np.sum(noise ** 2) / len(noise)

        # Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def apply_snr(self, signal, noise, mask=None):
        """Apply snr and augment signal

        Args:
            signal (np.array): original signal
            noise (np.array): noise signal
            mask (np.array): optional boolean mask of the samples the SNR is measured over (default: all)

        Returns:
            tuple: (augmented signal, scaled noise)
        """
        snr = self.config["snr"]
        s, n = (signal, noise) if mask is None else (signal[mask], noise[mask])
        # Calculate the power of the normalized original signal
        power_signal = np.sum(s ** 2) / len(s)

        # Calculate the power of the normalized noise
        power_noise = np.sum(n ** 2) / len(n)

        # Calculate the ratio of powers for the random SNR
        snr_ratio = 10 ** (snr / 10.0)

        # Calculate the required scaling factor for the noise to achieve the random SNR
        required_scaling_factor = np.sqrt(power_signal / (power_noise * snr_ratio))

        # Scale noise by the calculated factor to achieve the random SNR
        scaled_noise = noise * required_scaling_factor

        # Augment original signal with the scaled normalized noise at the random SNR
        augmented_signal = signal + scaled_noise
        applied_snr = self.calculate_snr(signal, scaled_noise, mask)
        if abs(applied_snr - snr) > 0.5:
            warnings.warn(f"Desired SNR and applied SNR differ more than 0.5")

        return augmented_signal, scaled_noise

    def select_files(self, rng=None):
        """pick: one file (the seeded cycle, or drawn from rng if given); sum: all of them."""
        if self.mix == "sum":
            return list(self.audio_files)
        if rng is None:
            return [next(self.random_audio_generator)]
        return [self.audio_files[int(rng.integers(len(self.audio_files)))]]

    def load_window(self, noise_filename, ts, sample_rate, rng=None):
        """Load a noise file, normalise it, and cut / pad it to ts samples.

        Returns:
            tuple: (noise window, offset in seconds of the window in the file (negative: zero-padded in front),
                    channel used or None for a downmix)
        """
        if self.channel is None:
            noise_signal, noise_sample_rate = librosa.load(noise_filename, sr=None)
            channel = None
        else:
            noise_signal, noise_sample_rate = librosa.load(noise_filename, sr=None, mono=False)
            noise_signal = np.atleast_2d(noise_signal)
            channel = self.channel
            if channel == "random":
                channel = (int(rng.integers(noise_signal.shape[0])) if rng is not None
                           else random.randrange(noise_signal.shape[0]))
            if channel >= noise_signal.shape[0]:
                raise ValueError(f"channel {channel} out of range for {noise_filename}")
            noise_signal = noise_signal[channel]

        # Resample the noise to match the sample rate of the audio data
        if noise_sample_rate != sample_rate:
            noise_signal = librosa.resample(noise_signal, orig_sr=noise_sample_rate, target_sr=sample_rate)

        noise = normalize_audio(noise_signal)
        draw = (lambda lo, hi: int(rng.integers(lo, hi + 1))) if rng is not None else random.randint
        tn = len(noise)  # Duration of the selected noise signal
        if ts <= tn:
            tn1 = draw(0, tn - ts)
            return noise[tn1:tn1 + ts], tn1 / sample_rate, channel
        if self.pad == "loop":
            start = draw(0, tn - 1)
            reps = int(np.ceil((start + ts) / tn))
            return np.tile(noise, reps)[start:start + ts], start / sample_rate, channel
        pad_front = draw(0, ts - tn)
        pad_end = ts - tn - pad_front
        return np.pad(noise, (pad_front, pad_end), mode='constant'), -pad_front / sample_rate, channel

    def run_with_info(self, audio_data, sample_rate, rng=None):
        """
        Add the noise and return what was applied.

        With rng=None every random choice comes from Python's random (the original behaviour); with a numpy
        Generator every choice (file, channel, window, pad position) comes from it.

        :return: tuple of the augmented audio and a dict with snr_db (measured on the scaled noise, with snr_mode),
            target_snr_db, noise_path ("|"-joined for mix "sum"), noise_offset_s, noise_channel, n_sources
        """
        files = self.select_files(rng)
        signal = normalize_audio(audio_data)
        ts = len(signal)  # Duration of the initial audio signal
        windows = [self.load_window(f, ts, sample_rate, rng) for f in files]
        if len(windows) == 1:
            noise = windows[0][0]
        else:  # equal-power sum: babble
            noise = np.sum([normalize_audio(w) for w, _, _ in windows], axis=0)

        mask = speech_mask(signal, sample_rate) if self.snr_mode == "speech" else None
        if mask is not None:
            whole_power = np.mean(noise ** 2)
            if np.mean(noise[mask] ** 2) < whole_power * 10 ** (-self.SILENT_NOISE_DB / 10):
                raise ValueError(f"The noise is silent during the speech frames ({'|'.join(files)})")

        # Apply the SNR and normalize the augmented signal
        s_aug, scaled_noise = self.apply_snr(signal, noise, mask)
        s_aug = s_aug / np.abs(s_aug).max()

        return s_aug, {
            "snr_db": float(self.calculate_snr(signal, scaled_noise, mask)),
            "target_snr_db": float(self.config["snr"]),
            "noise_path": "|".join(files),
            "noise_offset_s": "|".join(f"{o:.4f}" for _, o, _ in windows),
            "noise_channel": "|".join("mix" if c is None else str(c) for _, _, c in windows),
            "n_sources": len(files),
        }

    def run(self, audio_data, sample_rate):
        """
        Run the augmentation method

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate
        :return: tuple of the augmented audio data (numpy array) and the applied noise filename
        """
        s_aug, info = self.run_with_info(audio_data, sample_rate)
        return s_aug, self.applied_noise(info)

    def applied_noise(self, info):
        """The noise file name(s), as run() returns them."""
        return "|".join(os.path.basename(p) for p in info["noise_path"].split("|"))
