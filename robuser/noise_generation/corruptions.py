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

    def run(self, audio_data, sample_rate, output_file_path=None):
        """
        Run the augmentation method

        :param audio_data: numpy array with the audio data or the path to the audio file
        :param sample_rate: the sample rate
        :param output_file_path: the path to save the augmented audio data
        :return: tuple with the augmented audio data (numpy array) and the applied noise (or None)
        """
        raise NotImplementedError