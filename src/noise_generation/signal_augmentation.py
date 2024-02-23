"""Creating a signal augmented dataset.

There are four available augmentation methods:
    * compression
    * reverberation
    * clipping
    * gain

Maintainer Antonia Petrogianni
Copyright Behavioral Signals Technologies
"""

from audiomentations import (ApplyImpulseResponse, 
                             ClippingDistortion, 
                             GainTransition)

augmentation_mapping = {
    "ApplyImpulseResponse": "reverb",
    "ClippingDistortion": "clipping",
    "GainTransition": "gain"
}

class SignalAugmentation:
    def __init__(self, input_data:list, reverb_dataset:str, p_reverb=1.0,
                 min_gain_db=-24.0, max_gain_db=6.0,
                 min_gain_duration=1.0, max_gain_duration=3.0,
                 gain_duration_unit="seconds", p_gain=1.0, 
                 min_clipping_percentage=0, max_clipping_percentage=40, p_clipping=1.0):
        """
        Initialize the SignalAugmentation class

        :param input_data (list): list with the audio data
        :param reverb_dataset (str/Path): A path or list of paths to audio file(s) and/or folder(s) with audio files. 
        :param p_reverb (float): the probability of applying reverberation
        :param min_gain_db (float): the minimum gain in dB
        :param max_gain_db (float): the maximum gain in dB
        :param min_gain_duration (Union[float, int]): the minimum duration of the gain transition
        :param max_gain_duration (Union[float, int]): the maximum duration of the gain transition
        :param gain_duration_unit (str • choices: "fraction", "samples", "seconds"): 
            the unit of the value of min_gain_duration and max_gain_duration. 
                "fraction": Fraction of the total sound length
                "samples": Number of audio samples
                "seconds": Number of seconds
        :param p_gain (float): the probability of applying gain transition
        :param min_clipping_percentage (int): lower bound on the total percent of samples that will be clipped
        :param max_clipping_percentage (int): upper bound on the total percent of samples that will be clipped
        :param p_clipping (float): the probability of applying clipping distortion
        """

        self.input_data = input_data
        self.reverb_dataset = reverb_dataset
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.min_duration = min_gain_duration
        self.max_duration = max_gain_duration
        self.gain_duration_unit = gain_duration_unit
        self.min_clipping_percentage = min_clipping_percentage
        self.max_clipping_percentage = max_clipping_percentage

        self.p_reverb = p_reverb
        self.p_gain = p_gain    
        self.p_clipping = p_clipping

        if not self.input_data:
            raise ValueError("The input data is empty.")
        elif not isinstance(self.input_data, list):
            raise ValueError("The input data should be a list.")

    
    def apply_impulse_response(self):
        """
        Apply impulse response to the audio data
        
        :return: 
            the transform to be applied to the audio data
        """

        if self.reverb_dataset is None:
            raise ValueError("The `reverberation` dataset is not available.")
        
        transform = ApplyImpulseResponse(
            ir_path=self.reverb_dataset, 
            p=self.p_reverb
        )
        
        return transform

    def apply_clipping_distortion(self):
        """
        Apply clipping distortion to the audio data

        :return: 
            the transform to be applied to the audio data
        """
        
        transform = ClippingDistortion(
            min_percentile_threshold=self.min_clipping_percentage,
            max_percentile_threshold=self.max_clipping_percentage, 
            p=self.p_clipping
        )
        
        return transform
    
    def apply_gain_transition(self):
        """
        Apply gain transition to the audio data

        :return: 
            the transform to be applied to the audio data
        """
        
        transform = GainTransition(
            min_gain_db=self.min_gain_db,
            max_gain_db=self.max_gain_db,
            min_duration=self.min_gain_duration, 
            max_duration=self.max_gain_duration, 
            duration_unit=self.gain_duration_unit,
            p=self.p_gain
        )
        
        return transform
