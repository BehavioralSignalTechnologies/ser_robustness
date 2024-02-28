from corruptions import NoiseGeneration
from audiomentations import GainTransition

class AddGainTransition(NoiseGeneration):
    """
    Gradually change the volume up or down over a random time span. Also known as
    fade in and fade out. 

    config:    
        `min_duration`  (Union[float, int]): the minimum duration of the gain transition
        `max_duration`  (Union[float, int]): the maximum duration of the gain transition
        `duration_unit` ("fraction"): 
            the unit of the value of min_gain_duration and max_gain_duration. 
                "fraction": Fraction of the total sound length
        `p_gain` (float): the probability of applying gain transition
    """
    def __init__(self, config):
        """
        Initialize the GainTransition class

            :param config: dictionary with the configuration parameters
        """
        super().__init__(config)
        self.min_duration = config["duration"]
        self.max_duration = config["duration"]
        self.duration_unit = "fraction"
        self.p_gain = 1.0

    def run(self, audio_data, sample_rate):
        """
        Run the gain transition augmentation method

            :param audio_data: numpy array with the audio data
            :param sample_rate: the sample rate

            :return: the augmented audio data (numpy array)
        """
        transform = GainTransition(
            min_duration=self.min_duration, 
            max_duration=self.max_duration, 
            duration_unit=self.duration_unit,
            p=self.p_gain
        )

        return transform(audio_data, sample_rate), None