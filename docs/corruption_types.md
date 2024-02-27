# Noise Generation Types 🦾

In this document, we describe various types of noises and effects generated, including background noises, gain transition *(known as fade in/out)*, Gaussian noise, impulse response effects, clipping distortion and compression artifacts.

## Background Noises 🚦

Background noises are generated with a chosen Signal-to-Noise Ratio (SNR) value to simulate different environments.

### Datasets & SNRs:
- [**ESC50**](https://github.com/karoldvl/ESC-50/archive/master.zip) SNR: [0dB, 10dB, 20dB]
- [**MUSAN**](https://www.openslr.org/resources/17/musan.tar.gz) SNR: [0dB, 10dB, 20dB]
- [**UrbanSound8K** ](https://github.com/soundata/soundata#quick-example)SNR: [0dB, 10dB, 20dB]

> You can download `ESC50` and `MUSAN` from the links provided. To download `UrbanSound8K`, you can follow the commands below:

```
sudo apt-get install libasound2-dev
pip3 install soundata
```

```
python3
>> dataset = soundata.initialize('urbansound8k')
>> dataset.download()
```

## Gaussian Noise 💨

Gaussian noise is a kind of white noise that is added to the audio signal. It applies an SNR that is chosen randomly from a uniform distribution on the decibel scale.

### Different SNR values:
- 0dB
- 10dB
- 20dB

## Impulse Response Effects 🚇

Impulse responses are applied to simulate acoustic environments such as underground passages, underpasses, and stairwells from the [EchoThief Impulse Response Library](http://www.echothief.com/wp-content/uploads/2016/06/EchoThiefImpulseResponseLibrary.zip) dataset.

### Different types of reverberation
1. **Underground**: Simulates a reverberant effect of an underground environment.
2. **Underpasses**: Mimics the sound reflections found in underpasses.
3. **Stairwells**: Reproduces the reverberation of

## Clipping Distortion 📶

The signal is distorted by clipping a random percentage of points. The percentage of points that will be clipped is drawn from a uniform distribution between the two input parameters `min_percentile_threshold` and `max_percentile_threshold`. If for instance 30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.

### Clipping Distortion parameters:
- `min_percentile_threshold` (int): 0
- `max_percentile_threshold` (int): 40


## Gain Transition 〽️

Gain transition effect is applied to gradually change the volume up or down over a random time span. Also known as fade in and fade out.

### Gain Transition parameters:
- `min_gain_db`: the minimum gain in dB
- `max_gain_db`: the maximum gain in dB
- `min_duration` (Union[float, int]): the minimum duration of the gain transition
- `max_duration` (Union[float, int]): the maximum duration of the gain transition
- `duration_unit` (str • choices: "fraction", "seconds"): the unit of the value of `min_duration` and `max_duration` 

## Compression Artifacts

...