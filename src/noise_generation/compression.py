from pathlib import Path
from subprocess import run, DEVNULL
import os
import soundfile as sf


class Compression():
    """ A perturbator that compresses the audio file to a given bitrate using ffmpeg.
    The file is being compressed to the given format, then converted back to wav at
    the original sample rate with 1 channel.
    """

    def __init__(self, bitrate: int, compression_format: str = "mp3"):
        """ Initialize the perturbator.
        Args:
            bitrate: The bitrate to compress the audio to. The number you give will be
                transformed to <bitrate>k. E.g. 32 -> 32k -> 32000 bits per second.
            compression_format: The compression format to use.
        """

        self.bitrate = bitrate

        assert compression_format in ["mp3", "aac"] , f"format: {compression_format} is not supported"
        self.format = compression_format

    def perturb_audio(self, input_path: str, output_path: str):

        if not output_path.endswith(".wav"):
            print(f"WARNING: output_path: `{output_path}` doesn't have .wav suffix - "
                  f"Replacing {Path(output_path).suffix} with .wav")
            output_path = Path(output_path).with_suffix(".wav")

        assert not os.path.exists(output_path), f"output_path: {output_path} already exists. Specify a new path."

        compressed_path = Path(input_path).with_suffix(f".{self.format}")

        with sf.SoundFile(input_path, 'r') as audio_file:
            sample_rate = audio_file.samplerate

        compression_command = f"ffmpeg -i {input_path} -b:a {self.bitrate}k {compressed_path}"
        conversion_command = f"ffmpeg -i {compressed_path} -ac 1 -ar {sample_rate} {output_path}"

        run(f"{compression_command} && {conversion_command}", shell=True, check=False,
            stdout=DEVNULL, stderr=DEVNULL)
        
        # remove the .mp3 file
        os.remove(compressed_path)