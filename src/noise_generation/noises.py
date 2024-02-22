"""
Creating a noisy dataset.

There are four available augmentation methods:
    * reverberation
    * clipping
    * gain

Maintainer Antonia Petrogianni
Copyright Behavioral Signals Technologies
"""

import os
from audiomentations import (ApplyImpulseResponse,
                             GainTransition,
                             ClippingDistortion,
                             GaussianNoise
                             )


class NoiseGeneration:
    """
    NoiseGeneration class

    """

    def __init__(self, input_path:str, output_path:str, noise_type:str, noise_data:str):
        """
        Initialize the NoiseGeneration class

        :param input_path: input list of audio files
        :param output_path: output path for the noisy audio files
        :param noise_type: type of noise
        :param noise_data: noise data
        """
        self.input_path = input_path
        self.output_path = output_path
        self.noise_type = noise_type
        self.noise_data = noise_data

    def apply_gain_transition(self):
        """
        Apply gain transition to the signal

        :return: signal with gain transition
        """
        pass
    
    def apply_clipping_distortion(self):
        """
        Apply clipping distortion to the input data

        :return: 
        """
        max_clipping_percentage = 40
        augment = ClippingDistortion(max_percentile_threshold=self.max_clipping_percentage, p=1.0),
        pass

    def apply_impulse_response(self):
        """
        Apply impulse response to the input data

        :return: 
        """
        pass

    def apply_gaussian_noise(self):
        """
        Apply gaussian noise to the input data

        :return: 
        """
        pass

    def add_background_noise(self):
        """
        Add background noise to the input data

        :return: 
        """
        pass
    
                

