class CorruptionType:
    """
    CorruptionType base class

    """

    def __init__(self, config):
        """
        Initialize the CorruptionType class

        :param config: dictionary with the configuration parameters
        """
        self.config = config

    def run(self, audio_data, sample_rate):
        """
        Run the corruption method

        :param audio_data: numpy array with the audio data or the path to the audio file
        :param sample_rate: the sample rate of the audio data
        :return: tuple with the corrupted audio data (numpy array) and the applied noise (or None)
        """
        raise NotImplementedError

    def run_with_info(self, audio_data, sample_rate, rng=None):
        """
        Run the corruption and also return what was applied.

        Corruptions that support it override this to return the measured degradation (e.g. SNR, C50, SDR) and the
        parameters needed to reproduce the output. The default wraps run().

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate of the audio data
        :param rng: optional numpy Generator for the random choices; None keeps each corruption's original randomness
        :return: tuple with the corrupted audio data (numpy array) and a dict with the applied parameters/labels
        """
        audio, applied = self.run(audio_data, sample_rate)
        return audio, {"applied_noise_path": applied}

    def applied_noise(self, info):
        """What run() returns as the applied noise (e.g. the noise file or the impulse response), from the info that
        run_with_info returned. Corruptions that override run_with_info override this to match their run()."""
        return info.get("applied_noise_path")
