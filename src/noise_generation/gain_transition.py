"""
    Gradually change the volume up or down over a random time span. Also known as
    fade in and fade out. 

    config:    
        `min_gain_db`   (float): the minimum gain in dB
        `max_gain_db`   (float): the maximum gain in dB
        `min_duration`  (Union[float, int]): the minimum duration of the gain transition
        `max_duration`  (Union[float, int]): the maximum duration of the gain transition
        `duration_unit` (str • choices: "fraction", "seconds"): 
            the unit of the value of min_gain_duration and max_gain_duration. 
                "fraction": Fraction of the total sound length
                "seconds": Number of seconds
        `p_gain` (float): the probability of applying gain transition
"""

from noises import NoiseGeneration
from audiomentations import GainTransition

class AddGainTransition(NoiseGeneration):
    def __init__(self, config):
        """
        Initialize the GainTransition class

            :param config: dictionary with the configuration parameters
        """
        super().__init__(config)
        self.min_gain_db = config["min_gain_db"]
        self.max_gain_db = config["max_gain_db"]
        self.min_duration = config["min_duration"]
        self.max_duration = config["max_duration"]
        self.duration_unit = config["duration_unit"]
        self.p_gain = config["p_gain"]

    def run(self, audio_data, sample_rate):
        """
        Run the gain transition augmentation method

            :param audio_data: numpy array with the audio data
            :param sample_rate: the sample rate

            :return: the augmented audio data (numpy array)
        """
        transform = GainTransition(
            min_gain_db=self.min_gain_db,
            max_gain_db=self.max_gain_db,
            min_duration=self.min_duration, 
            max_duration=self.max_duration, 
            duration_unit=self.duration_unit,
            p=self.p_gain
        )

        return transform(audio_data, sample_rate), None

if __name__ == "__main__":
    import os, shutil, librosa
    from scipy.io import wavfile

    iemocap_dir = "/home/ubuntu/ser_robustness/iemocap/Session5/sentences/wav/"

    # find 3 random files from ieomcap_dir
    def foo():
        audio_files = []

        for root, dirs, files in os.walk(iemocap_dir):
            for file in files:
                if file.endswith(".wav"):
                    print(os.path.join(root, file))
                    audio_files.append(os.path.join(root, file))
                    if len(audio_files) == 5:
                        return audio_files


    audio_files = foo()

    config = {
        "min_gain_db": -24,
        "max_gain_db": 6,
        "min_duration": 1,
        "max_duration": 3,
        "duration_unit": "fraction",
        "p_gain": 1.0
    }

    augmentation = AddGainTransition(config)

    for audio_file in audio_files:
        shutil.copy(audio_file, ".")
        audio, sample_rate = librosa.load(audio_file, sr=None)
        s_aug, _ = augmentation.run(audio, sample_rate)
        basename = os.path.basename(audio_file)
        wavfile.write(f"{os.path.basename(audio_file).replace('.wav', '_fade.wav')}", sample_rate, s_aug)