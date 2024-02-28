import os
from corruptions import NoiseGeneration
from audiomentations import ApplyImpulseResponse


class AddImpulseResponse(NoiseGeneration):
    """
    Convolve the audio with a randomly selected impulse response.

    config: 
        `ir_path`  (str/Path): A path or list of paths to audio file(s) and/or folder(s) with audio files. 
        `p_reverb` (float): the probability of applying reverberation

    *download the echo thief impulse response dataset: http://www.echothief.com/downloads/
    """
    def __init__(self, config):
        """
        Initialize the ImpulseResponse class

            :param config: dictionary with the configuration parameters
        """
        super().__init__(config)
        self.ir_path = config["ir_path"]
        self.p_reverb = 1.0

        if self.ir_path is None:
            raise ValueError(f"The {self.ir_path} is not provided."+
                             "Please provide the path to the impulse response files.")

        if not os.path.exists(self.ir_path):
            raise ValueError(f"The {self.ir_path} does not exist.")
        
        self.transform = ApplyImpulseResponse(
            ir_path=self.ir_path, 
            p=self.p_reverb
        )

    def run(self, audio_data, sample_rate):
        """
        Run the impulse response method

            :param audio_data: numpy array with the audio data
            :param sample_rate: the sample rate

            :return: the augmented audio data (numpy array)
        """

        return self.transform(audio_data, sample_rate), None