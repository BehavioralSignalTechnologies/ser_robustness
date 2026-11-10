from pathlib import Path
import tempfile
import subprocess
from subprocess import run, DEVNULL
import os

import numpy as np
import soundfile as sf

from robuser.corruptions.corruption_type import CorruptionType


class Compression(CorruptionType):
    """ A perturbator that compresses the audio file to a given bit_rate using ffmpeg.
    The file is being compressed to the given format, then converted back to the original
    audio format at the original sample rate with 1 channel.

    Codecs (ffmpeg; the output depends on the ffmpeg / libopus build):
        * mp3 (default): the original behaviour
        * mulaw, alaw: G.711 (64 kbps), encoded at 8 kHz
        * g726: G.726 ADPCM (16 / 24 / 32 / 40 kbps), encoded at 8 kHz (ffmpeg's encoder only accepts 8 kHz)
        * opus: Opus in VoIP mode (6-64 kbps), at the input's rate (8 / 12 / 16 / 24 / 48 kHz)
    G.711 / G.726 are encoded at 8 kHz and decoded back to the input's rate, as on a phone line. The new codecs return
    exactly the input's length.
    """

    CODECS = ("mp3", "mulaw", "alaw", "g726", "opus")
    BIT_RATES = {"mp3": range(8, 193), "mulaw": (64,), "alaw": (64,), "g726": (16, 24, 32, 40), "opus": range(6, 65)}
    PHONE_RATE = 8000

    def __init__(self, config):
        """ Initialize the perturbator.
        Args:
            config
                codec: one of mp3 (default), mulaw, alaw, g726, opus
                bit_rate: The bit_rate to compress the audio to. The number you give will be
                transformed to <bit_rate>k. E.g. 32 -> 32k -> 32000 bits per second.
                Optional for mulaw / alaw (64).
        """
        super().__init__(config)
        self.codec = config.get("codec", "mp3")
        if self.codec not in self.CODECS:
            raise ValueError(f"`codec` must be one of {self.CODECS}")
        self.format = "mp3"
        self.bit_rate = config.get("bit_rate", 64 if self.codec in ("mulaw", "alaw") else None)

        if self.codec == "mp3":
            if not isinstance(self.bit_rate, int) or self.bit_rate < 8 or self.bit_rate > 192:
                raise ValueError("`bit_rate` must be an integer between 8 and 192kHz.")
        elif self.bit_rate not in self.BIT_RATES[self.codec]:
            valid = self.BIT_RATES[self.codec]
            valid = f"{valid.start}-{valid.stop - 1}" if isinstance(valid, range) else ", ".join(map(str, valid))
            raise ValueError(f"`bit_rate` for {self.codec} must be one of: {valid} (kbps)")

    @staticmethod
    def ffmpeg(args, data):
        """Run ffmpeg on stdin -> stdout, raising with its error output if it fails."""
        result = subprocess.run(["ffmpeg", "-v", "error", *args], input=data, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({' '.join(args)}): {result.stderr.decode(errors='replace').strip()}")
        return result.stdout

    def run_codec(self, audio_data, sample_rate):
        """Encode / decode with a telephony / VoIP codec through pipes; same length as the input."""
        pcm = (np.clip(audio_data, -1, 1) * 32767).astype("<i2").tobytes()
        raw = ["-f", "s16le", "-ar", str(sample_rate), "-ac", "1"]
        if self.codec in ("mulaw", "alaw"):
            enc = ["-ar", str(self.PHONE_RATE), "-c:a", f"pcm_{self.codec}", "-f", self.codec]
            dec_in = ["-f", self.codec, "-ar", str(self.PHONE_RATE), "-ac", "1"]
        elif self.codec == "g726":
            enc = ["-ar", str(self.PHONE_RATE), "-c:a", "adpcm_g726", "-b:a", f"{self.bit_rate}k", "-f", "wav"]
            dec_in = []
        else:
            enc = ["-c:a", "libopus", "-b:a", f"{self.bit_rate}k", "-application", "voip", "-f", "ogg"]
            dec_in = []
        bits = self.ffmpeg([*raw, "-i", "-", *enc, "-"], pcm)
        out = self.ffmpeg([*dec_in, "-i", "-", *raw, "-"], bits)
        y = (np.frombuffer(out, "<i2").astype(np.float32) / 32767).astype(audio_data.dtype)
        if len(y) < len(audio_data):
            y = np.pad(y, (0, len(audio_data) - len(y)))
        return y[: len(audio_data)]

    def run_mp3(self, audio_data, sample_rate):
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

            return compressed_audio

        finally:
            # Clean up temporary files
            for temp_file in [temp_input_path, temp_compressed_path, temp_output_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def run_with_info(self, audio_data, sample_rate, rng=None):
        """Compress and return the codec and bit_rate (nothing is random)."""
        out = self.run_mp3(audio_data, sample_rate) if self.codec == "mp3" else self.run_codec(audio_data, sample_rate)
        return out, {"codec": self.codec, "bit_rate": self.bit_rate}

    def run(self, audio_data, sample_rate):
        return self.run_with_info(audio_data, sample_rate)[0], None
