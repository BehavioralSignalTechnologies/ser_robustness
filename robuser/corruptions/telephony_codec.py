import audioop  # standard library up to Python 3.12 (this package requires ~=3.12)
import subprocess

import numpy as np

from robuser.corruptions.corruption_type import CorruptionType


class TelephonyCodec(CorruptionType):
    """
    Encode and decode the audio with a telephony / VoIP codec at its native rate (e.g. 8 kHz).

    Unlike `compression` (MP3), these are the codecs speech actually travels through on calls:
        * mulaw, alaw: G.711 (64 kbps)
        * g726: G.726 ADPCM, bitrate from `bitrates` (default 16 / 24 / 32 kbps)
        * opus: Opus, VoIP mode, bitrate from `bitrates` (default 6 / 8 / 12 / 16 kbps)
    G.726 and Opus need ffmpeg on the PATH.

    config:
        * codec: one of mulaw, alaw, g726, opus
        * bitrates: optional list of kbps to draw from (g726 / opus); the bitrate is drawn from the rng
    """

    CODECS = ("mulaw", "alaw", "g726", "opus")
    DEFAULT_BITRATES = {"g726": (16, 24, 32), "opus": (6, 8, 12, 16)}

    def __init__(self, config):
        super().__init__(config)
        if config.get("codec") not in self.CODECS:
            raise ValueError(f"codec must be one of {self.CODECS}")

    def run_with_info(self, audio_data, sample_rate, rng=None, context=None):
        rng = rng if rng is not None else np.random.default_rng()
        codec = self.config["codec"]
        pcm = (np.clip(audio_data, -1, 1) * 32767).astype("<i2").tobytes()
        if codec == "mulaw":
            out, kbps = audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2), 64
        elif codec == "alaw":
            out, kbps = audioop.alaw2lin(audioop.lin2alaw(pcm, 2), 2), 64
        else:
            kbps = int(rng.choice(list(self.config.get("bitrates", self.DEFAULT_BITRATES[codec]))))
            if codec == "g726":
                enc = ["-c:a", "adpcm_g726", "-b:a", f"{kbps}k", "-f", "wav"]
            else:
                enc = ["-c:a", "libopus", "-b:a", f"{kbps}k", "-application", "voip", "-f", "ogg"]
            raw = ["-f", "s16le", "-ar", str(sample_rate), "-ac", "1"]
            bits = subprocess.run(["ffmpeg", "-v", "error", *raw, "-i", "-", *enc, "-"],
                                  input=pcm, capture_output=True, check=True).stdout
            out = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", *raw, "-"],
                                 input=bits, capture_output=True, check=True).stdout
        y = np.frombuffer(out, "<i2").astype(np.float32) / 32767
        if len(y) < len(audio_data):
            y = np.pad(y, (0, len(audio_data) - len(y)))
        return y[: len(audio_data)], {"codec": codec, "codec_kbps": kbps}

    def run(self, audio_data, sample_rate):
        audio, _ = self.run_with_info(audio_data, sample_rate)
        return audio, None
