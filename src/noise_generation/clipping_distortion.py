"""
    Distort signal by clipping a random percentage of points

        :param min_percentile_threshold: lower bound on the total percent of samples that will be clipped
        :param max_percentile_threshold: upper bound on the total percent of samples that will be clipped

    The percentage of points that will be clipped is drawn from a uniform distribution between
    the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
    30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.
"""

from noises import NoiseGeneration
from audiomentations import ClippingDistortion

class AddClippingDistortion(NoiseGeneration):
    def __init__(self, config):
        """
        Initialize the ClippingDistortion class

            :param config: dictionary with the configuration parameters
        """
        super().__init__(config)
        self.min_percentile_threshold = config["min_percentile_threshold"]
        self.max_percentile_threshold = config["max_percentile_threshold"]
        self.p_clipping = config["p_clipping"]

    def run(self, audio_data, sample_rate):
        """
        Run the clipping distortion augmentation method

            :param audio_data: numpy array with the audio data
            :param sample_rate: the sample rate

            :return: the augmented audio data (numpy array)
        """
        transform = ClippingDistortion(
            min_percentile_threshold=self.min_percentile_threshold,
            max_percentile_threshold=self.max_percentile_threshold, 
            p=self.p_clipping
        )
        return transform(audio_data, sample_rate), None
