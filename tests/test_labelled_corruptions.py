"""The labels returned by run_with_info are the applied amounts, a seed fixes the output, and level keeps the labels."""

import numpy as np
import pytest
import soundfile as sf

from robuser.corruptions.chain import Chain
from robuser.corruptions.clipping_distortion import AddClippingDistortion
from robuser.corruptions.content import ContentCorruption
from robuser.corruptions.impulse_response import AddImpulseResponse, calculate_c50
from robuser.corruptions.level import Level
from robuser.corruptions.utils import sdr_db, speech_mask

SR = 8000


def speech_like(seconds=4.0, seed=0):
    """Harmonic bursts (syllables) separated by near-silence, so the speech mask is not all ones."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = (np.sin(2 * np.pi * 1.5 * t) > 0.2).astype(float)
    x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((180, 360, 540, 900), 1))
    return (0.3 * env * x + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture
def noise_file(tmp_path):
    path = tmp_path / "noise.wav"
    stereo = np.random.default_rng(1).standard_normal((3 * SR, 2)) * [0.1, 0.3]
    sf.write(path, stereo, SR)
    return str(path)


@pytest.fixture
def rir_file(tmp_path):
    """Exponential decay (RT60 ~0.4 s) after a 50 ms lead-in."""
    rng = np.random.default_rng(2)
    n, lead = int(0.6 * SR), int(0.05 * SR)
    h = np.zeros(n)
    h[lead] = 1.0
    tail = rng.standard_normal(n - lead - 1) * np.exp(-6.9 * np.arange(n - lead - 1) / (0.4 * SR)) * 0.3
    h[lead + 1:] = tail
    path = tmp_path / "rir.wav"
    sf.write(path, h, SR)
    return str(path)


@pytest.mark.parametrize("snr_mode", ["whole", "speech"])
@pytest.mark.parametrize("snr", [-5.0, 7.3, 30.0])
def test_content_measured_snr_is_target(noise_file, snr_mode, snr):
    content = ContentCorruption({"content_dataset_path": noise_file, "snr": snr, "pad": "loop", "snr_mode": snr_mode,
                                 "channel": "random"})
    x = speech_like(seconds=5.0)  # longer than the noise: looped
    out, info = content.run_with_info(x, SR, rng=np.random.default_rng(0))
    assert info["snr_db"] == pytest.approx(snr, abs=1e-3)
    assert len(out) == len(x) and np.abs(out).max() == pytest.approx(1.0)


def test_content_babble_sums_every_file(tmp_path):
    files = []
    for i in range(3):
        files.append(str(tmp_path / f"talker{i}.wav"))
        sf.write(files[-1], speech_like(seconds=3.0, seed=i) * (i + 1), SR)
    content = ContentCorruption({"content_dataset_path": files, "snr": 5.0, "mix": "sum", "snr_mode": "speech"})
    _, info = content.run_with_info(speech_like(), SR, rng=np.random.default_rng(0))
    assert info["n_sources"] == 3 and info["noise_path"] == "|".join(sorted(files))
    assert info["snr_db"] == pytest.approx(5.0, abs=1e-3)


def test_content_noise_silent_during_speech_raises(tmp_path):
    x = speech_like()
    pauses = ~speech_mask(x, SR)
    noise = (np.random.default_rng(3).standard_normal(len(x)) * 0.1 * pauses).astype(np.float32)  # only in pauses
    path = tmp_path / "pauses.wav"
    sf.write(path, noise, SR)
    content = ContentCorruption({"content_dataset_path": str(path), "snr": 10.0, "snr_mode": "speech"})
    with pytest.raises(ValueError, match="silent"):
        content.run_with_info(x, SR, rng=np.random.default_rng(0))


@pytest.mark.parametrize("clip_type", ["hard", "asym", "soft"])
@pytest.mark.parametrize("sdr", [3.0, 15.0, 35.0])
def test_clipping_measured_sdr_is_target(clip_type, sdr):
    clipping = AddClippingDistortion({"sdr": sdr, "clip_type": clip_type, "asym_ratio": 0.5})
    x = speech_like().astype(np.float64)
    out, info = clipping.run_with_info(x, SR)
    assert sdr_db(x, out) == pytest.approx(sdr, abs=0.05)
    assert info["sdr_db"] == pytest.approx(sdr_db(x, out))


def test_impulse_response_c50_and_alignment(rir_file):
    x = np.random.default_rng(4).standard_normal(4 * SR).astype(np.float32)
    ir = AddImpulseResponse({"ir_path": rir_file, "channel": 0, "align": True})
    out, info = ir.run_with_info(x, SR, rng=np.random.default_rng(0))
    assert info["c50_db"] == pytest.approx(calculate_c50(rir_file, 0, SR)[0])
    assert np.isfinite(info["c50_db"]) and 0.2 < info["rt60_s"] < 0.6
    lag = np.argmax(np.correlate(out[:4000], x[:4000], "full")) - 3999  # aligned: the output is not delayed
    assert abs(lag) <= int(0.008 * SR)
    unaligned, _ = AddImpulseResponse({"ir_path": rir_file, "channel": 0}).run_with_info(x, SR)
    lag = np.argmax(np.correlate(unaligned[:4000], x[:4000], "full")) - 3999
    assert lag == pytest.approx(0.05 * SR, abs=2)  # the 50 ms lead-in


def chain_config(seed, noise_file):
    return {"seed": seed, "steps": [
        {"type": "content", "content_dataset_path": noise_file, "snr": 10.0, "pad": "loop", "snr_mode": "speech",
         "channel": "random"},
        {"type": "clipping_distortion", "sdr": 12.0},
        {"type": "level", "mode": "rms"}]}


def test_same_seed_same_output(noise_file):
    x = speech_like()
    a, info_a = Chain(chain_config(7, noise_file)).run_with_info(x, SR)
    b, info_b = Chain(chain_config(7, noise_file)).run_with_info(x, SR)
    c, _ = Chain(chain_config(8, noise_file)).run_with_info(x, SR)
    assert np.array_equal(a, b) and info_a == info_b
    assert not np.array_equal(a, c)
    assert info_a["0_content_snr_db"] == pytest.approx(10.0, abs=1e-3)
    assert info_a["1_clipping_distortion_sdr_db"] == pytest.approx(12.0, abs=0.05)


@pytest.mark.parametrize("mode", ["rms", "peak"])
def test_level_leaves_earlier_labels_unchanged(mode):
    x = speech_like().astype(np.float64)
    clipped, info = AddClippingDistortion({"sdr": 10.0}).run_with_info(x, SR)
    level = Level({"mode": mode})
    out, level_info = level.run_with_info(clipped, SR, rng=np.random.default_rng(0))
    gain = 10 ** (level_info["gain_db"] / 20)
    assert sdr_db(x * gain, out) == pytest.approx(info["sdr_db"], abs=1e-6)
    assert level_info["peak"] <= 0.99 + 1e-9
    assert out.dtype == x.dtype


def test_per_file_applied_noise_paths_match_run(tmp_path, noise_file, rir_file):
    """corrupt_dataset_per_file reports the same applied noise as run(): the noise file name for content, the impulse
    response path for impulse_response, None for the others."""
    import csv
    import json
    import random

    from robuser.dataset_corruption.corrupt_dataset_per_file import apply_corruption_from_csv

    clean = tmp_path / "clean.wav"
    sf.write(clean, speech_like(), SR)
    specs = {"content": {"content_dataset_path": noise_file, "snr": 10},
             "impulse_response": {"ir_path": rir_file}, "clipping_distortion": {"max_percentile_threshold": 20}}
    plan = tmp_path / "plan.csv"
    with open(plan, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_file_path", "corruption_type", "corruption_metadata", "output_file_path"])
        for t, meta in specs.items():
            w.writerow([clean, t, json.dumps(meta), tmp_path / f"{t}.wav"])
    random.seed(0)
    applied = apply_corruption_from_csv(str(plan), force=True)
    assert applied[str(tmp_path / "content.wav")] == "noise.wav"
    assert applied[str(tmp_path / "impulse_response.wav")] == rir_file
    assert applied[str(tmp_path / "clipping_distortion.wav")] is None
