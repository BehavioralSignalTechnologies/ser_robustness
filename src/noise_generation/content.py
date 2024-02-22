from noises import NoiseGeneration

import soundfile as sf


class ContentAugmentation(NoiseGeneration):
    """
    ContentAugmentation class
    """

    def __init__(self, config):
        """
        Initialize the ContentAugmentation class

        :param config: dictionary with the configuration parameters
        """
        super().__init__(config)

    def run(self, audio_data, sample_rate):
        """
        Run the augmentation methods

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate
        :return: the augmented audio data (numpy array)
        """
        raise NotImplementedError


if __name__ == "__main__":
    config = {

    }
    augmentation = ContentAugmentation(config)

    audio, sample_rate = sf.read("Ses05F_impro01/Ses05F_impro01_F000.wav")
    augmentation.run(audio, sample_rate)
