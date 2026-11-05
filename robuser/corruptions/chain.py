import random

import numpy as np

from robuser.corruptions.corruption_type import CorruptionType


class Chain(CorruptionType):
    """
    Apply several corruptions in order, as one reproducible step, and return what each of them applied.

    All random choices come from one seed: the steps that take an rng (content, impulse_response, level, ...) draw
    from one generator, consumed in order, and Python's and numpy's global generators are seeded too (on the chain's
    first run), for the steps that draw from those (audiomentations' parameters, gaussian). So a row of the per-file
    CSV fully determines its output. In batch mode the chain keeps drawing from the same generator, so every file
    gets its own random choices and the whole run is reproducible. Each step's output is cast back to the chain's
    dtype.

    Put `level` before `compression`: compression goes through 16-bit audio, so anything above 1 is clipped there.

    config:
        * seed: integer seed (used when run_with_info gets no rng, e.g. when the chain is run directly)
        * steps: list of {"type": <corruption name>, ...that corruption's config}
        * dtype: working precision, "float32" (default) or "float64"
        * name: optional name (batch mode output folder); default: the step types joined with "+"
    The returned info keys are prefixed with the step's position and type, e.g. "1_content_snr_db".
    Example (reverberant speech with mild noise, at a random level, through Opus):
        {"seed": 7, "steps": [{"type": "impulse_response", "ir_path": "room.wav", "channel": 0, "align": true},
                              {"type": "content", "content_dataset_path": "cafe.wav", "snr": 25, "pad": "loop"},
                              {"type": "level", "mode": "rms"},
                              {"type": "compression", "codec": "opus", "bit_rate": 12}]}
    """

    def __init__(self, config):
        super().__init__(config)
        from robuser.corruptions.get_corruption import get_corruption  # here to avoid a circular import

        if "seed" not in config or "steps" not in config:
            raise ValueError("chain needs a seed and a list of steps")
        self.dtype = np.dtype(config.get("dtype", "float32"))
        if self.dtype not in (np.float32, np.float64):
            raise ValueError("dtype must be float32 or float64")
        self.steps = []
        for step in config["steps"]:
            params = {k: v for k, v in dict(step).items() if k != "type"}
            self.steps.append((step["type"], get_corruption(step["type"])(params)))
        self.name = config.get("name") or "+".join(name for name, _ in self.steps)
        self.rng = np.random.default_rng(int(config["seed"]))
        self.global_seeded = False

    def run_with_info(self, audio_data, sample_rate, rng=None):
        seed = int(self.config["seed"])
        rng = rng if rng is not None else self.rng
        if not self.global_seeded:
            random.seed(seed)
            np.random.seed(seed)
            self.global_seeded = True
        audio = np.asarray(audio_data, dtype=self.dtype)
        info = {"seed": seed, "steps": "|".join(name for name, _ in self.steps)}
        for i, (name, corruption) in enumerate(self.steps):
            audio, step_info = corruption.run_with_info(audio, sample_rate, rng=rng)
            audio = np.asarray(audio, dtype=self.dtype)
            info.update({f"{i}_{name}_{k}": v for k, v in step_info.items()})
        return audio, info

    def run(self, audio_data, sample_rate):
        audio, _ = self.run_with_info(audio_data, sample_rate)
        return audio, None
