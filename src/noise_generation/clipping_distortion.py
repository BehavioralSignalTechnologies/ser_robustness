"""
    Distort signal by clipping a random percentage of points

        :param min_percentile_threshold: lower bound on the total percent of samples that will be clipped
        :param max_percentile_threshold: upper bound on the total percent of samples that will be clipped

    The percentage of points that will be clipped is drawn from a uniform distribution between
    the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
    30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.
"""

from corruptions import NoiseGeneration
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
        "min_percentile_threshold": 0,
        "max_percentile_threshold": 40,
        "p_clipping": 1.0
    }

    augmentation = AddClippingDistortion(config)

    for audio_file in audio_files:
        shutil.copy(audio_file, ".")
        audio, sample_rate = librosa.load(audio_file, sr=None)
        s_aug, _ = augmentation.run(audio, sample_rate)
        basename = os.path.basename(audio_file)
        wavfile.write(f"{os.path.basename(audio_file).replace('.wav', '_distortion.wav')}", sample_rate, s_aug)