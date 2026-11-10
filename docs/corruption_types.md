# Corruption Types 🦾

Here, we describe the various types of corruptions applied to the clean dataset and the parameters that can be adjusted to customize the corruption process.
For a detailed explanation of the configuration file, refer to the [Configuration](configuration.md) section.


## Background Noises 🚦

Background noises are generated with a chosen Signal-to-Noise Ratio (SNR) value to simulate different environments.
Configure the `content` section in the `config.yml` file to add background noise to the clean dataset.

### Download the datasets

- [**ESC50**](https://github.com/karoldvl/ESC-50/archive/master.zip)
- [**MUSAN**](https://www.openslr.org/resources/17/musan.tar.gz)
- [**UrbanSound8K** ](https://github.com/soundata/soundata#quick-example)

> To download `ESC50` and `MUSAN` use the provided links. For `UrbanSound8K`, follow the instructions below:

```
sudo apt-get install libasound2-dev
pip3 install soundata
```

```
python3
>> dataset = soundata.initialize('urbansound8k')
>> dataset.download()
```

### Default SNR values:
- 0dB
- 10dB
- 20dB

### Options (defaults keep the behaviour above)

| key | values | default | |
|---|---|---|---|
| `content_dataset_path` | a folder, a file, or a list of files / folders | | a single file is used as is (e.g. the noise a plan chose for a file) |
| `pad` | `zero` / `loop` | `zero` | `loop` repeats a noise shorter than the audio, so it covers the whole file |
| `snr_mode` | `whole` / `speech` | `whole` | `speech`: SNR over the speech frames of the input (energy VAD), so the level does not depend on how much silence the file has. Raises if the noise is silent during the speech frames |
| `mix` | `pick` / `sum` | `pick` | `sum`: every file of `content_dataset_path`, each normalised to equal power, summed (babble) |
| `channel` | omitted / index / `random` | omitted (downmix) | a multichannel noise contributes one channel instead of the average of its channels |

Returned by `run_with_info`: `snr_db` (measured on the scaled noise, with `snr_mode`), `target_snr_db`, `noise_path`
(`|`-joined for `sum`), `noise_offset_s` (negative: zero-padded in front), `noise_channel`, `n_sources`.

## Gaussian Noise 💨

Add White Gaussian Noise to the clean dataset with different Signal-to-Noise Ratio (SNR) levels. Configure the `gaussian` section in the `config.yml` file to add Gaussian noise to the clean dataset.

### Default SNR values:
- 10dB
- 20dB
- 30dB

## Impulse Response Effects 🚇

Impulse responses are applied to emulate acoustic environments like underground passages, underpasses, stairwells, etc.
A random impulse response within a given range of RT60 values from the dataset is convolved with the clean audio to simulate the effect of the environment on the sound.
Configure the `impulse_response` section in the `config.yml` file to apply impulse response effects to the clean dataset.

### Download the dataset
- [**EchoThief Impulse Response Library**](http://www.echothief.com/wp-content/uploads/2016/06/EchoThiefImpulseResponseLibrary.zip)

You can resample the dataset to match your data sample rate.
Otherwise, the impulse responses will be resampled on the fly, potentially hurting execution time.
Additionally, note that the RT60 values of the dataset will be calculated on the dataset without resampling and it may affect their values.
If you are unsure about your dataset's sample rate or there are multiple sample rates across audios, do nothing and let the code resample on the fly.

### Default RT60 range values:
- 0.1-0.5 seconds
- 0.5-1.0 seconds
- 1.0-1.5 seconds

### Options (defaults keep the behaviour above)

| key | values | default | |
|---|---|---|---|
| `ir_path` | a folder, a file, or a list of files / folders | | |
| `rt60_range` | `[min, max]` | required for a folder | optional for files (no filtering) |
| `channel` | omitted / index | omitted (downmix) | apply one channel of a multichannel response (e.g. a microphone array): the sum of responses at different microphones is not a physical response |
| `align` | `true` / `false` | `false` | cut the response where its direct sound starts (8 ms before the direct-path peak), so the output is not delayed by the response's lead-in and C50 describes exactly the applied response |

`calculate_c50(path, channel, sample_rate)` returns the C50 clarity and the direct-to-reverberant ratio (DRR):
C50 = 10 log10(E[t0 − 8 ms, t0 + 50 ms) / E[t0 + 50 ms, end)), DRR = 10 log10(E[t0 − 8 ms, t0 + 8 ms) / E[t0 + 8 ms, end)),
t0 = the direct-path peak (± 8 ms reproduces the ACE corpus DRR values). Compute it at the sample rate of the audio the
response is applied to: the values depend on the bandwidth. `calculate_rt60` takes the same optional arguments.

Returned by `run_with_info`: `ir_path`, `channel`, `rt60_s`, `c50_db`, `drr_db`, computed on the applied channel at the
audio's sample rate.

## Clipping Distortion 📶

The signal is distorted by clipping a random percentage of points. The percentage of points to be clipped is drawn from a uniform distribution between 0 and `max_percentile_threshold`. For instance, if 30% is drawn, samples below the 15th or above the 85th percentile are clipped.
Configure the `clipping_distortion` section in the `config.yml` file to apply clipping distortion to the clean dataset.

### Default max_percentile_threshold values:
- 20
- 40
- 60

### Target SDR

With `sdr` (dB) the clipped share is not drawn: it is solved (bisection) so that the signal-to-distortion ratio
SDR = 10 log10(Σ x² / Σ (x − clipped x)²) hits the target, and `max_percentile_threshold` is not needed.

| key | values | default | |
|---|---|---|---|
| `sdr` | dB | omitted (drawn share, as above) | |
| `clip_type` | `hard` / `asym` / `soft` | `hard` | `hard`: the clipping above; `asym`: the clipped share is split unevenly between the two sides; `soft`: tanh saturation, its threshold solved for the target SDR |
| `asym_ratio` | (0, 1] | 1 | `asym`: the less clipped side's share as a fraction of the other's |
| `asym_neg` | `true` / `false` | `true` | `asym`: the negative side is the more clipped one |

Returned by `run_with_info`: `sdr_db` (measured, in both modes), `clipped_pct`, `clip_type`, and with `sdr` also
`target_sdr_db` and the solved `percentile` with `lower_threshold` / `upper_threshold` (× the peak) (hard / asym),
or `theta` (soft, × the peak).


## Gain Transition 〽️

Gain transition effect is applied to gradually change the volume up or down over a time span (aka fade in/out). It applies a gain transition to the clean audio. The gain starts at the first gain and is held constant until the transition starts, and it transitions to the second gain. Then, that gain is held constant until the end of the sound. Configure the `gain_transition` section in the `config.yml` file to apply gain transition to the clean dataset.


### Default min_max_gain_db:
- [-40.0, -20.0]
- [-30.0, -10.0]
- [-20.0, 0.0]

> lists of [min_gain_db, max_gain_db] pairs

## Compression Artifacts 🎛️

Compression artifacts are generated by compressing an audio file to a given bit_rate using `ffmpeg`. The file is being compressed to an .mp3 format, then converted back to wav at the original sample rate with 1 channel. Configure the `compression` section in the `config.yml` file to apply compression artifacts to the clean dataset.
    

### Default bit_rate values:
- 8kbps
- 16kbps
- 32kbps

### Codecs

| `codec` | `bit_rate` (kbps) | |
|---|---|---|
| `mp3` (default) | 8-192 | as above |
| `mulaw`, `alaw` | 64 (optional) | G.711, encoded at 8 kHz |
| `g726` | 16 / 24 / 32 / 40 | G.726 ADPCM, encoded at 8 kHz (ffmpeg's encoder only accepts 8 kHz) |
| `opus` | 6-64 | Opus, VoIP mode, at the audio's rate (8 / 12 / 16 / 24 / 48 kHz) |

G.711 / G.726 are encoded at 8 kHz and decoded back to the audio's rate, as on a phone line. The new codecs return
exactly the input's length and raise on ffmpeg errors (e.g. an ffmpeg built without libopus). The output depends on the
ffmpeg / libopus build (tested: ffmpeg 6.1, libopus 1.3.1). Put `level` before `compression` in a chain: the audio goes
through 16 bits, so anything above 1 is clipped. Returned by `run_with_info`: `codec`, `bit_rate`.

## Level 🔊

Sets the output level: a random RMS level (`mode: rms`, `rms_db: [min, max]` dBFS, default [-35, -15]) or the peak
near full scale (`mode: peak`, `peak: [min, max]`, default [0.90, 0.99]). It scales the whole signal, so the labels
returned by earlier steps (SNR, SDR, C50) are unchanged. The peak never exceeds `max_peak` (0.99), so in `rms` mode a
file with a high peak-to-RMS ratio can end up quieter than the drawn target: the measured level is returned. Draws from
the rng it is given, or from its own generator (`seed`, default 42). Configure the `level` section in `config.yml`
(disabled by default).

Returned by `run_with_info`: `mode`, `target_rms_dbfs` or `target_peak`, `gain_db`, `rms_dbfs`, `peak`.

## Chain ⛓️

Applies several corruptions in order, from one seed: the steps that take an rng draw from one generator, consumed in
order, and Python's and numpy's global generators are seeded too, so a row of the per-file CSV fully determines its
output. Each step's output is cast back to the chain's `dtype` (`float32` default, or `float64`). The returned keys are
prefixed with the step's position and type, e.g. `0_content_snr_db`.

Example per-file CSV row (reverberant speech with mild noise, at a random level, through Opus):

```
audio_file_path,corruption_type,corruption_metadata,output_file_path
clean.wav,chain,"{""seed"": 7, ""steps"": [{""type"": ""impulse_response"", ""ir_path"": ""room.wav"", ""channel"": 0, ""align"": true}, {""type"": ""content"", ""content_dataset_path"": ""cafe.wav"", ""snr"": 25, ""pad"": ""loop"", ""snr_mode"": ""speech""}, {""type"": ""level"", ""mode"": ""rms""}, {""type"": ""compression"", ""codec"": ""opus"", ""bit_rate"": 12}]}",out/reverb.wav
```

In batch mode (`config.yml`, disabled by default) `steps` is a list of step lists, one chain each, and the output folder
is named by `name` (default: the step types joined with `+`). There the chain keeps drawing from its generator across
files, so each file gets its own random choices.


> ℹ️ For the majority of the corruptions, we use the [audiomentations](https://github.com/iver56/audiomentations.git) library.

# Labels: what was applied 🏷️

Every corruption returns what it applied through `run_with_info(audio, sample_rate, rng=None)` (the fields are listed in
each section above), so its outputs can be used as regression targets. With `rng=None` a corruption keeps its original
randomness; with a numpy `Generator` every random choice comes from it. The per-file mode (`corrupt_dataset_per_file`)
writes the fields to `--info_output`, one row per output file; an optional `"seed"` in a row's `corruption_metadata`
makes that row reproducible on its own.

```
python3 -m robuser.dataset_corruption.corrupt_dataset_per_file -i plan.csv --info_output info.csv -w 16
```
