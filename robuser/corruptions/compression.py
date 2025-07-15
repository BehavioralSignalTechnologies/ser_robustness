from pathlib import Path
import tempfile
from subprocess import run, DEVNULL
import os
import soundfile as sf

from robuser.corruptions.corruption_type import CorruptionType


class Compression(CorruptionType):
    """ A perturbator that compresses the audio file to a given bit_rate using ffmpeg.
    The file is being compressed to the given format, then converted back to the original
    audio format at the original sample rate with 1 channel.
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

    def run(self, audio_data, sample_rate):
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix=f'.{self.format}', delete=False) as temp_compressed:
            temp_compressed_path = temp_compressed.name
            
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Write audio data to temporary input file
            sf.write(temp_input_path, audio_data, sample_rate)
            
            # Compress the audio
            # Overwrite the output file
            compression_command = f"ffmpeg -y -i {temp_input_path} -b:a {self.bit_rate}k {temp_compressed_path}"
            conversion_command = f"ffmpeg -y -i {temp_compressed_path} -ac 1 -ar {sample_rate} {temp_output_path}"
            
            run(f"{compression_command} && {conversion_command}", 
                shell=True, check=False,
                stdout=DEVNULL, stderr=DEVNULL)
            
            # Read the compressed audio back
            compressed_audio, _ = sf.read(temp_output_path)
            
            return compressed_audio, None
            
        finally:
            # Clean up temporary files
            for temp_file in [temp_input_path, temp_compressed_path, temp_output_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
