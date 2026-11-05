import os
import random
import numpy as np
import librosa
import soundfile as sf
import warnings


def get_supported_audio_extensions():
    """
    Returns a tuple of supported audio file extensions.
    
    Returns:
        tuple: Supported audio file extensions in lowercase
    """
    return (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma")


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
    audio_extensions = get_supported_audio_extensions()

    # Iterate through the original folder to find the target sample rate
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(audio_extensions):
                _, target_sr = librosa.load(os.path.join(root, file), sr=None)
                break
        if target_sr:
            break

    # Iterate through the resampled folder to resample files
    for root, dirs, files in os.walk(resampled_folder_path):
        for file in files:
            if file.lower().endswith(audio_extensions):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)

                if target_sr != sr:
                    if not need_resampling:
                        need_resampling = True
                        warnings.warn(f"Resampling from {sr} to {target_sr}...")

                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                    sf.write(file_path, y, target_sr)


def audio_files_in(paths, sort=True):
    """Audio files of a path or a list of paths: a folder is walked, a file is taken as is.

    Returns:
        tuple: (list of audio files, sorted unless sort=False (os.walk order), True if any of the paths is a folder)
    """
    paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    audio_extensions = get_supported_audio_extensions()
    files, any_folder = [], False
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        if os.path.isdir(path):
            any_folder = True
            for root, _, names in os.walk(path):
                files.extend(os.path.join(root, n) for n in names if n.lower().endswith(audio_extensions))
        else:
            files.append(str(path))
    return (sorted(files) if sort else files), any_folder


def load_channel(path, sample_rate, channel=None, rng=None):
    """Load an audio file at sample_rate (librosa). channel: None = downmix to mono, an int = that channel,
    "random" = a channel drawn from rng (or Python's random if rng is None).

    Returns:
        tuple: (signal, channel used, or None for a downmix)
    """
    if channel is None:
        y, _ = librosa.load(path, sr=sample_rate, mono=True)
        return y, None
    y, _ = librosa.load(path, sr=sample_rate, mono=False)
    y = np.atleast_2d(y)
    if channel == "random":
        channel = int(rng.integers(y.shape[0])) if rng is not None else random.randrange(y.shape[0])
    if not 0 <= int(channel) < y.shape[0]:
        raise ValueError(f"channel {channel} out of range for {path} ({y.shape[0]} channels)")
    return y[int(channel)], int(channel)


def speech_mask(signal, sample_rate, frame_s=0.02, db_below_peak=35.0):
    """Sample-level speech mask (energy VAD): 20 ms frames within 35 dB of the 95th-percentile frame energy."""
    frame = int(round(frame_s * sample_rate))
    n = len(signal) // frame
    energy = 10 * np.log10(np.mean(signal[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    active = energy > (np.percentile(energy, 95) - db_below_peak)
    mask = np.zeros(len(signal), bool)
    mask[: n * frame] = np.repeat(active, frame)
    return mask


def sdr_db(clean, distorted):
    """Signal-to-distortion ratio in dB of `distorted` w.r.t. `clean` (inf if they are identical)."""
    d = np.sum((np.asarray(clean, float) - np.asarray(distorted, float)) ** 2)
    return float("inf") if d <= 0 else float(10 * np.log10(np.sum(np.asarray(clean, float) ** 2) / d))
