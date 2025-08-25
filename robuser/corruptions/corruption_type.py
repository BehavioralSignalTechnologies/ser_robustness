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
