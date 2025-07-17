import os
import random

import librosa
import pyroomacoustics as pra
from audiomentations import ApplyImpulseResponse

from robuser.corruptions.corruption_type import CorruptionType
from robuser.corruptions.utils import get_supported_audio_extensions


def calculate_rt60(impulse_response_path):
    """
    Calculate the RT60 of the impulse response
        :param impulse_response_path: the path to the impulse response
        :return: the RT60 in seconds
    """
    impulse_response, sample_rate = librosa.load(impulse_response_path, sr=None)
    # Normalize the impulse response
    norm_impulse_response = impulse_response / max(abs(impulse_response))
    rt60 = pra.experimental.measure_rt60(norm_impulse_response, fs=sample_rate)
    return rt60


class AddImpulseResponse(CorruptionType):
    """
    Convolve the audio with a randomly selected impulse response.

    config: 
        `ir_path`  (str/Path): A path or list of paths to audio file(s) and/or folder(s) with audio files. 
        `rt60_range` (float, float): The range of the RT60 in seconds of the impulse responses to be used.

    *download the echo thief impulse response dataset: http://www.echothief.com/downloads/
    """

    def __init__(self, config):
        """
        Initialize the ImpulseResponse class

            :param config: dictionary with the configuration parameters
        """
        super().__init__(config)

        if "ir_path" not in config:
            raise ValueError("The ir_path is not provided. " +
                             "Please provide the path to the impulse response files.")

        if "rt60_range" not in config:
            raise ValueError("The rt60_range is not provided. " +
                             "Please provide the range of RT60 in seconds.")
        
        self.ir_path = config["ir_path"]

        if not os.path.exists(self.ir_path):
            raise ValueError(f"The {self.ir_path} does not exist.")
        
        if not isinstance(config["rt60_range"], (list, tuple)) or len(config["rt60_range"]) != 2:
            raise ValueError("rt60_range must be a list or tuple of [min_rt60, max_rt60]")
        
        self.rt60_min, self.rt60_max = config["rt60_range"]

        random.seed(42)
        self.selected_irs = list(self.load_dataset(self.ir_path, rt60_min=self.rt60_min, rt60_max=self.rt60_max))
        print(f"Selected {len(self.selected_irs)} impulse responses from {self.ir_path}"
              f" with RT60 in range [{self.rt60_min}, {self.rt60_max}]")

    def load_dataset(self, path, rt60_min, rt60_max):
        audio_extensions = get_supported_audio_extensions()
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    rt60 = calculate_rt60(os.path.join(root, file))
                    if rt60_min <= rt60 <= rt60_max:
                        yield os.path.join(root, file)

    def run(self, audio_data, sample_rate):
        """
        Run the impulse response method

            :param audio_data: numpy array with the audio data
            :param sample_rate: the sample rate

            :return: the augmented audio data (numpy array) and the applied impulse response
        """
        ir_wav_path = random.choice(self.selected_irs)
        transform = ApplyImpulseResponse(
            ir_path=ir_wav_path,
            p=1.0
        )
        return transform(audio_data, sample_rate), ir_wav_path
