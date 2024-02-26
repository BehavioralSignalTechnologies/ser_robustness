import numpy as np


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
