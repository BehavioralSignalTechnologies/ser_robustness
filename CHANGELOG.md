# Changelog

All notable changes to RobuSER. Versions follow [semantic versioning](https://semver.org/); every pull request that
changes behaviour bumps the version and adds an entry here. Install a version with
`uv pip install "git+https://github.com/BehavioralSignalTechnologies/ser_robustness.git@v<version>"`.

## [0.2.0]

Corruptions can now return what they applied, so their outputs can be used as regression targets (e.g. the exact SNR,
C50 or SDR of a file), and several corruptions can run in one reproducible step. The defaults of the existing
corruptions are unchanged.

### Added
- `run_with_info(audio, sample_rate, rng=None)` on every corruption: returns the output and a dict with what was
  applied. With `rng=None` each corruption keeps its original randomness; with a numpy `Generator` every random choice
  comes from it. `corrupt_dataset_per_file` writes these dicts to `--info_output`, accepts an optional `seed` in a
  row's `corruption_metadata` (the row is then reproducible on its own) and can run in parallel (`--workers`).
- `content`: `content_dataset_path` also takes a file or a list of files / folders; `pad: zero | loop`;
  `snr_mode: whole | speech` (SNR over the speech frames of the input); `mix: pick | sum` (sum = equal-power babble of
  every file); `channel` (omitted = downmix, an index, or `random`). Returns `snr_db` (measured), `target_snr_db`,
  `noise_path`, `noise_offset_s`, `noise_channel`, `n_sources`. With `snr_mode: speech` it raises if the noise is
  silent during the speech frames.
- `impulse_response`: `ir_path` also takes a file or a list; `rt60_range` is optional for files; `channel` (apply one
  channel of a multichannel response instead of the downmix); `align` (cut the response where its direct sound
  starts). New `calculate_c50` (C50 and DRR) next to `calculate_rt60`, both with optional `channel` and
  `sample_rate`. Returns `ir_path`, `channel`, `rt60_s`, `c50_db`, `drr_db`.
- `clipping_distortion`: `sdr` (the clipped share is solved for a target signal-to-distortion ratio instead of drawn),
  `clip_type: hard | asym | soft`. Returns `sdr_db` (measured, also in the default mode), `clipped_pct`, and with
  `sdr` the solved `percentile` / `theta`.
- `compression`: `codec: mp3 | mulaw | alaw | g726 | opus` (G.711 / G.726 encoded at 8 kHz, Opus in VoIP mode); the
  new codecs raise on ffmpeg errors. Returns `codec`, `bit_rate`.
- `level`: a new corruption that sets the output level (random RMS, or peak near full scale).
- `chain`: a new corruption that applies several corruptions in order from one seed.
- `utils`: `speech_mask` (energy VAD), `sdr_db`, `audio_files_in`, `load_channel`.
- Tests (`tests/`).

### Fixed
- `content` normalised its output by `max(x)` instead of `max(|x|)`, so a file whose negative peak was larger than its
  positive one went above 1 and clipped when saved. **This changes the paper's `content` outputs for those files**
  (about half of the files in our check); use v0.1.0 to reproduce the paper exactly.

### Notes
- `gaussian` draws its noise from numpy's global generator without a seed, so its output differs from run to run
  (in 0.1.0 too). Inside a `chain` the global generators are seeded.

## [0.1.0]

The state used for the paper (commit `175694c`): the batch (`corrupt_dataset`) and per-file
(`corrupt_dataset_per_file`) corruption of a dataset with `content`, `gaussian`, `gain_transition`,
`clipping_distortion`, `impulse_response` and `compression`, the IEMOCAP parser, and the evaluation scripts.
