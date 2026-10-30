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

    def run_with_info(self, audio_data, sample_rate, rng=None, context=None):
        """
        Run the corruption and also return what was applied.

        Labelled corruptions override this to return the measured degradation (e.g. SNR, C50, SDR) and every
        parameter needed to reproduce the output. The default wraps run() for the existing corruptions.

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate of the audio data
        :param rng: numpy Generator for any random choice (labelled corruptions are deterministic given it)
        :param context: dict shared by the steps of a chain (e.g. the speech mask of the clean input)
        :return: tuple with the corrupted audio data (numpy array) and a dict with the applied parameters/labels
        """
        audio, applied = self.run(audio_data, sample_rate)
        return audio, {"applied_noise_path": applied}
