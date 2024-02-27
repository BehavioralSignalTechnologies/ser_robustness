"""
    Convolve the audio with a randomly selected impulse response.

    config: 
        `ir_path`  (str/Path): A path or list of paths to audio file(s) and/or folder(s) with audio files. 
        `p_reverb` (float): the probability of applying reverberation

    *download the echo thief impulse response dataset: http://www.echothief.com/downloads/
"""
import os
from corruptions import NoiseGeneration
from audiomentations import ApplyImpulseResponse


class AddImpulseResponse(NoiseGeneration):
    def __init__(self, config):
        """
        Initialize the ImpulseResponse class

            :param config: dictionary with the configuration parameters
        """
        super().__init__(config)
        self.ir_path = config["ir_path"]
        self.p_reverb = config["p_reverb"]

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

        return self.transform(audio_data, sample_rate), os.path.basename(self.ir_path)


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
        "ir_path": "/home/ubuntu/ser_robustness/EchoThiefImpulseResponseLibrary/Underground",
        "p_reverb": 1.0
    }

    augmentation = AddImpulseResponse(config)

    for audio_file in audio_files:
        shutil.copy(audio_file, ".")
        audio, sample_rate = librosa.load(audio_file, sr=None)
        s_aug, _ = augmentation.run(audio, sample_rate)
        basename = os.path.basename(audio_file)
        wavfile.write(f"{os.path.basename(audio_file).replace('.wav', '_reverb.wav')}", sample_rate, s_aug)