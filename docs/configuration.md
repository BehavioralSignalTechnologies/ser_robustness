# 📝 Configuration YAML File Guide

This guide provides information about the configuration YAML file (`config.yaml`) used for the audio corruptions.

## 🏗️ Structure

The `config.yaml` file follows a structured format with different sections for each type of audio corruption. Here's a breakdown of its structure:

```yaml
- content:                         # add background noise
  - enabled:                       # if the process is enabled or not.
  - content_dataset_path:          # Paths to different datasets.

- gaussian:                        # insert gaussian noise
  - enabled:                       # if the process is enabled or not.
  - snr:                           # Signal-to-noise ratio levels.

- gain_transition:                 # gradually change volume up/down
  - enabled:                       # if the process is enabled or not
  - min_max_gain_db:               # list of min-max gain dB pairs

- clipping_distortion:             # distort signal by clipping samples
  - enabled:                       # if the process is enabled or not
  - max_percentile_threshold:      # An upper bound on the total 
                                   # percent of samples that will         
                                   # be clipped. The lower bound is 0.

- impulse_response:                # Convolves the audio with an 
                                   # impulse response file
  - enabled:                       # if the process is enabled or not
  - ir_path:                       # Path to impulse response dataset.
  - rt60_range:                    # list of different ranges of 
                                   # reverberation times.

- compression:                     # compresses an audio file to a 
                                   # given bit_rate (from wav->mp3) and 
                                   # is converted back to wav 
                                   # at its original bit_rate
  - enabled:                       # if the process is enabled or not
  - bit_rate:                      # bit rate
```
## How to Use

1. **Enable/Disable Corruption**: Set the `enabled` field to `true` or `false` to enable or disable specific corruption type.

2. **Specify Parameters**: Adjust the parameters under each corrutpion type according to your requirements. For example, you can specify different SNR levels for Gaussian noise or set different bit rates for compressing an audio. Edit the `config.yaml` accordingly.


## 📚 Example

```yaml
content:
  enabled: true
  content_dataset_path:
    - datasets/ESC-50
    - datasets/urbansound8k
    - datasets/musan
gaussian:
  enabled: true
  snr:
    - 0
    - 10
    - 20
gain_transition:
  enabled: true
  min_max_gain_db: 
    - [-40.0,-20.0]
    - [-30.0,-10.0]
    - [-20.0,0.0]
# ...
