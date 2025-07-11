import os
import numpy as np
import librosa
import soundfile as sf
import warnings


def normalize_audio(signal):
    """A function to normalize a signal according to its mean and std.

    Args:
        signal (np.array): signal

    Returns:
        np.array: normalized signal
    """
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std


def resample_dataset(folder_path, resampled_folder_path):
    """A function to get the target sample rate from a folder of audio files 
       and resample all the files to the target sample rate.

    Args:
        folder_path (str): path to the folder with target audio files 
        resampled_folder_path (str): path to the folder with the files to be resampled
    """

    target_sr = None
    need_resampling = False

    # Iterate through the original folder to find the target sample rate
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                _, target_sr = librosa.load(os.path.join(root, file), sr=None)
                break
        if target_sr:
            break

    # Iterate through the resampled folder to resample files
    for root, dirs, files in os.walk(resampled_folder_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)

                if target_sr != sr:
                    if not need_resampling:
                        need_resampling = True
                        warnings.warn(f"Resampling from {sr} to {target_sr}...")

                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                    sf.write(file_path, y, target_sr)
