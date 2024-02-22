# ESC50 dataset: https://github.com/karolpiczak/ESC-50
# UrbanSound8K dataset: Use the script here: https://github.com/soundata/soundata#quick-example
# MUSAN dataset: https://www.openslr.org/resources/17/musan.tar.gz
import os

from noises import NoiseGeneration

import soundfile as sf


class ContentAugmentation(NoiseGeneration):
    """
    ContentAugmentation class
    config should contain:
        * content_dataset_path: the path to the dataset
        * content_dataset_name: the name of the dataset (ESC50, UrbanSound8K, MUSAN)
        * SNR: the signal to noise ratio
    """

    def __init__(self, config):
        """
        Initialize the ContentAugmentation class

        :param config: dictionary with the configuration parameters
        """
        super().__init__(config)
        if "content_dataset_path" not in config:
            raise ValueError("content_dataset_path is not in the config")
        if "content_dataset_name" not in config:
            raise ValueError("content_dataset_name is not in the config")
        if "SNR" not in config:
            raise ValueError("SNR is not in the config")

        self.dataset_path = config["content_dataset_path"]
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist")

        self.audio_files = self.get_audio_files()


    def get_audio_files(self):
        """
        Get the audio files from the dataset
        Returns:
            a sorted list with the audio files
        """
        audio_files = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(root, file))

        if len(audio_files) != len(set([os.path.basename(file) for file in audio_files])):
            raise ValueError("There are duplicate filenames in the dataset")

        return sorted(audio_files)

    def run(self, audio_data, sample_rate):
        """
        Run the augmentation methods

        :param audio_data: numpy array with the audio data
        :param sample_rate: the sample rate
        :return: the augmented audio data (numpy array)
        """
        raise NotImplementedError


if __name__ == "__main__":
    config = {
        "content_dataset_path": "/data_drive/ESC-50-master",
        "content_dataset_name": "ESC50",
        "SNR": 10
    }
    augmentation = ContentAugmentation(config)
    import pdb; pdb.set_trace()

    # audio, sample_rate = sf.read("Ses05F_impro01/Ses05F_impro01_F000.wav")
    # augmentation.run(audio, sample_rate)
