"""Acoustic labels (C50, DRR, T60) of room impulse responses.

Labels are computed on the RIR at the sample rate of the audio it is applied to (e.g. 8 kHz), i.e. on the same
band-limited response that is convolved with speech; they differ from full-band values.

Definitions (t0 = direct-path peak = argmax |h|, w = 8 ms):
  C50 = 10 log10( E[t0-w, t0+50ms) / E[t0+50ms, end) )
  DRR = 10 log10( E[t0-w, t0+w)    / E[t0+w, end) )    (+-8 ms reproduces the ACE corpus DRR values)
  T60 = Schroeder EDC (after noise-floor subtraction), line fit on -5..-25 dB (T20) x3,
        falling back to -5..-15 dB (T10) x6 if the decay does not reach -25 dB.
Validated on the ACE corpus at 48 kHz: T60 r = 0.98 (MAE 56 ms), DRR identical to the published values.

Usage (all channels of every .wav under a folder):
  python -m robuser.rir_labels --input /path/to/rirs --output rir_labels.csv --sample_rate 8000
"""

import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

from robuser.corruptions.signal_utils import resample

DIRECT_WIN_S = 0.008
C50_S = 0.050


def load_rir(path, sample_rate, channel=0):
    """One channel of an RIR file, resampled to sample_rate (float64)."""
    h, sr = sf.read(path, dtype="float64", always_2d=True)
    return resample(h[:, channel], sr, sample_rate)


def energy_ratio_db(num, den):
    return float(10 * np.log10(num / den)) if den > 0 and num > 0 else float("nan")


def decay_time(h, sample_rate, t0):
    """Return (T60, method, decay_range_db) from the Schroeder energy decay curve."""
    x = h[t0:] ** 2
    if len(x) < sample_rate * 0.05:
        return float("nan"), "too_short", float("nan")
    # noise floor from the last 10% of the response; subtract it before integrating
    tail = x[int(len(x) * 0.9) :]
    noise = float(np.mean(tail))
    x_clean = np.clip(x - noise, 0, None)
    edc = np.cumsum(x_clean[::-1])[::-1]
    if edc[0] <= 0:
        return float("nan"), "no_energy", float("nan")
    edc_db = 10 * np.log10(np.maximum(edc / edc[0], 1e-12))
    t = np.arange(len(edc_db)) / sample_rate
    # usable decay range: before EDC collapses to the clipped floor
    valid = edc_db > -60
    reach = float(edc_db[valid].min()) if valid.any() else 0.0
    for lo, hi, name in ((-5, -25, "T20"), (-5, -15, "T10")):
        if reach <= hi:
            m = (edc_db <= lo) & (edc_db >= hi)
            if m.sum() >= 5:
                slope = np.polyfit(t[m], edc_db[m], 1)[0]  # dB / s
                if slope < 0:
                    return float(-60.0 / slope), name, reach
    return float("nan"), "insufficient_decay", reach


def labels_for(h, sample_rate):
    """C50, DRR, T60 (+ diagnostics) of one RIR channel at sample_rate."""
    t0 = int(np.argmax(np.abs(h)))
    w = int(round(DIRECT_WIN_S * sample_rate))
    n50 = int(round(C50_S * sample_rate))
    e = h**2
    start = max(0, t0 - w)
    c50 = energy_ratio_db(e[start : t0 + n50].sum(), e[t0 + n50 :].sum())
    drr = energy_ratio_db(e[start : t0 + w].sum(), e[t0 + w :].sum())
    t60, method, reach = decay_time(h, sample_rate, t0)
    n_tail = int(sample_rate * 0.02)  # how far the tail had decayed when the file ends (truncation check)
    end_level = (
        energy_ratio_db(e[-n_tail:].mean(), e[t0 : t0 + n_tail].mean()) if len(e) > t0 + n_tail else float("nan")
    )
    return dict(c50_db=c50, drr_db=drr, t60_s=t60, t60_method=method, edc_reach_db=reach,
                peak_s=t0 / sample_rate, len_s=len(h) / sample_rate, end_level_db=end_level)


def _label_one(args):
    path, channel, sample_rate = args
    try:
        return {"path": path, "channel": channel, **labels_for(load_rir(path, sample_rate, channel), sample_rate),
                "error": ""}
    except Exception as ex:  # keep going; report per file
        return {"path": path, "channel": channel, "error": repr(ex)}


def main():
    ap = argparse.ArgumentParser(description="Compute C50 / DRR / T60 for every RIR channel under a folder")
    ap.add_argument("--input", required=True, help="Folder searched recursively for .wav files")
    ap.add_argument("--output", required=True, help="Output CSV")
    ap.add_argument("--sample_rate", type=int, default=8000, help="Rate at which the labels are computed")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.input, "**", "*.wav"), recursive=True))
    jobs = [(f, ch, a.sample_rate) for f in files for ch in range(sf.info(f).channels)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_label_one, jobs, chunksize=64))
    fields = ["path", "channel", "c50_db", "drr_db", "t60_s", "t60_method", "edc_reach_db", "peak_s", "len_s",
              "end_level_db", "error"]
    with open(a.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {a.output}: {len(rows)} RIR channels, {sum(bool(r['error']) for r in rows)} errors")


if __name__ == "__main__":
    main()
