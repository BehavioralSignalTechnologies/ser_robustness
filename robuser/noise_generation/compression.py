from pathlib import Path
from subprocess import run, DEVNULL
import os
import soundfile as sf

from robuser.noise_generation.corruptions import NoiseGeneration


class Compression(NoiseGeneration):
    """ A perturbator that compresses the audio file to a given bit_rate using ffmpeg.
    The file is being compressed to the given format, then converted back to wav at
    the original sample rate with 1 channel.
    """

    def __init__(self, config):
        """ Initialize the perturbator.
        Args:
            config
                bit_rate: The bit_rate to compress the audio to. The number you give will be
                transformed to <bit_rate>k. E.g. 32 -> 32k -> 32000 bits per second.
        """
        super().__init__(config)
        self.format = "mp3"
        self.bit_rate = config["bit_rate"]

        if not isinstance(self.bit_rate, int) or self.bit_rate < 8 or self.bit_rate > 192:
            raise ValueError("`bit_rate` must be an integer between 8 and 192kHz.")

    def run(self, audio_data, sample_rate, output_file_path):
        compressed_path = Path(audio_data).with_suffix(f".{self.format}")

        compression_command = f"ffmpeg -i {audio_data} -b:a {self.bit_rate}k {compressed_path}"
        conversion_command = f"ffmpeg -i {compressed_path} -ac 1 -ar {sample_rate} {output_file_path}"

        run(f"{compression_command} && {conversion_command}", 
            shell=True, check=False,
            stdout=DEVNULL, stderr=DEVNULL)
        
        os.remove(compressed_path)
