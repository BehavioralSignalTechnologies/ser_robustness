from audiomentations import AddGaussianSNR

from noises import NoiseGeneration


class AWGNAugmentation(NoiseGeneration):
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
        self.snr = config["snr"]

        self.transform = AddGaussianSNR(min_snr_in_db=self.snr, max_snr_in_db=self.snr, p=1.0)

    def run(self, audio_data, sample_rate):
        """
        Run the augmentation method

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate
        :return: the augmented audio data (numpy array)
        """
        return self.transform(audio_data, sample_rate)


if __name__ == "__main__":
    import os, shutil, librosa
    from scipy.io import wavfile

    iemocap_dir = "/data_drive/iemocap/Session5/sentences/wav/"


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

    for snr in [0, 5, 10, 20]:
        config = {
            "snr": snr
        }
        augmentation = AWGNAugmentation(config)

        for audio_file in audio_files:
            shutil.copy(audio_file, ".")
            audio, sample_rate = librosa.load(audio_file, sr=None)
            s_aug = augmentation.run(audio, sample_rate)
            basename = os.path.basename(audio_file)
            wavfile.write(f"augmented_{basename}_snr_{snr}.wav", sample_rate, s_aug)
