from audiomentations import AddGaussianSNR
from robuser.corruptions.corruption_type import CorruptionType


class AWGNAugmentation(CorruptionType):
    """
    Class that augments the audio data with additive white Gaussian noise so that the signal-to-noise ratio (SNR) is
    equal to the specified value.
    config should contain:
        * snr: the signal-to-noise ratio
    """

    def __init__(self, config):
        super().__init__(config)

        if "snr" not in config:
            raise ValueError("SNR is not in the config")
        if not isinstance(config["snr"], int):
            raise ValueError("SNR must be an integer")

        self.snr = config["snr"]

        self.transform = AddGaussianSNR(min_snr_in_db=self.snr, max_snr_in_db=self.snr, p=1.0)

    def run(self, audio_data, sample_rate):
        """
        Run the augmentation method

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate
        :return: the augmented audio data (numpy array)
        """
        return self.transform(audio_data, sample_rate), None
