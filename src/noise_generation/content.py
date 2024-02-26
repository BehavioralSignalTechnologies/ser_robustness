# ESC50 dataset: https://github.com/karolpiczak/ESC-50
# UrbanSound8K dataset: Use the script here: https://github.com/soundata/soundata#quick-example
# MUSAN dataset: https://www.openslr.org/resources/17/musan.tar.gz

import os
import random
import shutil
import warnings
from itertools import cycle

import librosa
import numpy as np
from scipy.io import wavfile

from noises import NoiseGeneration
from utils import normalize_audio


class ContentCorruption(NoiseGeneration):
    """
    Class that augments the audio data with content from a sound dataset
    config should contain:
        * content_dataset_path: the path to the dataset
        * snr: the signal-to-noise ratio
    """

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

        self.dataset_path = config["content_dataset_path"]
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist")

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
        audio_files = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(root, file))

        if len(audio_files) != len(set([os.path.basename(file) for file in audio_files])):
            raise ValueError("There are duplicate filenames in the dataset")

        return sorted(audio_files)

    def calculate_snr(self, signal, noise):
        """Calculates the snr from signal and noise

        Args:
            signal (np.array): clean signal
            noise (np.array): added noise

        Returns:
            float: snr
        """
        # Calculate the power of the signal
        signal_power = np.sum(signal ** 2) / len(signal)

        # Calculate the power of the noise
        noise_power = np.sum(noise ** 2) / len(noise)

        # Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def apply_snr(self, signal, noise):
        """Apply snr and augment signal

        Args:
            signal (np.array): original signal
            noise (np.array): noise signal

        Returns:
            np.array: augmented signal
        """
        snr = self.config["snr"]
        # Calculate the power of the normalized original signal
        power_signal = np.sum(signal ** 2) / len(signal)

        # Calculate the power of the normalized noise
        power_noise = np.sum(noise ** 2) / len(noise)

        # Calculate the ratio of powers for the random SNR
        snr_ratio = 10 ** (snr / 10.0)

        # Calculate the required scaling factor for the noise to achieve the random SNR
        required_scaling_factor = np.sqrt(power_signal / (power_noise * snr_ratio))

        # Scale noise by the calculated factor to achieve the random SNR
        scaled_noise = noise * required_scaling_factor

        # Augment original signal with the scaled normalized noise at the random SNR
        augmented_signal = signal + scaled_noise
        applied_snr = self.calculate_snr(signal, scaled_noise)
        if abs(applied_snr - snr) > 0.5:
            warnings.warn(f"Desired SNR and applied SNR differ more than 0.5")

        return augmented_signal

    def run(self, audio_data, sample_rate):
        """
        Run the augmentation method

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate
        :return: tuple of the augmented audio data (numpy array) and the applied noise filename
        """

        # Load a random noise from the dataset
        noise_filename = next(self.random_audio_generator)
        noise_basename = os.path.basename(noise_filename)
        noise_signal, noise_sample_rate = librosa.load(noise_filename, sr=None)

        # Resample the noise to match the sample rate of the audio data
        if noise_sample_rate != sample_rate:
            noise_signal = librosa.resample(noise_signal, orig_sr=noise_sample_rate, target_sr=sample_rate)

        # Normalize the audio data and the noise
        signal = normalize_audio(audio_data)
        noise = normalize_audio(noise_signal)
        ts = len(signal)  # Duration of the initial audio signal
        tn = len(noise)  # Duration of the selected noise signal
        if ts <= tn:
            tn1 = random.randint(0, tn - ts)
            tn2 = tn1 + ts
            noise = noise[tn1:tn2]
        else:
            pad_front = random.randint(0, ts - tn)
            pad_end = ts - tn - pad_front
            noise = np.pad(noise, (pad_front, pad_end), mode='constant')

        # Apply the SNR and normalize the augmented signal
        s_aug = self.apply_snr(signal, noise)
        s_aug = s_aug / np.abs(s_aug.max())

        return s_aug, noise_basename
