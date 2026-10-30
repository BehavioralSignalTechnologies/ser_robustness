import numpy as np

from robuser.corruptions.corruption_type import CorruptionType
from robuser.corruptions.signal_utils import speech_mask


class Chain(CorruptionType):
    """
    Apply several corruptions in order, as one reproducible step, and merge what they return.

    All random choices (noise window, output level, codec bitrate, ...) come from one generator seeded with `seed`,
    consumed by the steps in order, so a row of the per-file CSV fully determines its output. The speech mask of the
    clean input is computed once and shared, so every SNR in the chain is measured over the same speech frames.

    config:
        * seed: integer seed of the chain's random generator
        * steps: list of {"type": <corruption name>, ...that corruption's config}
        * dtype: working precision, "float32" (default) or "float64"
    Example (reverberant speech with mild noise, at a random level, through Opus):
        {"seed": 7, "steps": [{"type": "rir", "rir_path": "room.wav", "c50_cap": 30},
                              {"type": "noise_snr", "noise_path": "cafe.wav", "snr": 25, "min_noise_power": 0},
                              {"type": "level", "mode": "rms"}, {"type": "telephony_codec", "codec": "opus"}]}
    """

    def __init__(self, config):
        super().__init__(config)
        from robuser.corruptions.get_corruption import get_corruption  # here to avoid a circular import

        if "seed" not in config or "steps" not in config:
            raise ValueError("chain needs a seed and a list of steps")
        self.steps = []
        for step in config["steps"]:
            params = {k: v for k, v in dict(step).items() if k != "type"}
            self.steps.append((step["type"], get_corruption(step["type"])(params)))

    def run_with_info(self, audio_data, sample_rate, rng=None, context=None):
        rng = np.random.default_rng(int(self.config["seed"]))
        audio = np.asarray(audio_data, dtype=self.config.get("dtype", "float32"))
        context = {"speech_mask": speech_mask(audio, sample_rate)}
        info = {"steps": "|".join(name for name, _ in self.steps), "seed": int(self.config["seed"]),
                "dur_s": len(audio) / sample_rate, "speech_frac": float(context["speech_mask"].mean())}
        for _, corruption in self.steps:
            audio, step_info = corruption.run_with_info(audio, sample_rate, rng=rng, context=context)
            info.update(step_info)
        return audio, info

    def run(self, audio_data, sample_rate):
        audio, _ = self.run_with_info(audio_data, sample_rate)
        return audio, None
