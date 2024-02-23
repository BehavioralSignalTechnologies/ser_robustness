"""
Creating a noisy dataset.

There are four available augmentation methods:
    * reverberation
    * clipping
    * gain

Maintainer Antonia Petrogianni
Copyright Behavioral Signals Technologies
"""


class NoiseGeneration:
    """
    NoiseGeneration base class

    """

    def __init__(self, config):
        """
        Initialize the NoiseGeneration class

        :param config: dictionary with the configuration parameters
        """
        self.config = config

    def run(self, audio_data, sample_rate):
        """
        Run the augmentation method

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate
        :return: tuple with the augmented audio data (numpy array) and the applied noise (or None)
        """
        raise NotImplementedError
