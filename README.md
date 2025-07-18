# 💪 ROBUSER

A robustness evaluation benchmarking procedure for Speech Emotion Recognition (SER) 💬

## 💁 Installation guidelines

- Install FFmpeg & libasound2-dev:

```
sudo apt install ffmpeg libasound2-dev
```

- Install using uv

```
uv venv -p python3.12
source .venv/bin/activate
uv sync --active
```

> The dependencies have been installed 👏

## 📰 Documentation

Before running the dataset corruption scripts you first need to download the
required noise datasets and configure the respective parameters. For detailed
documentation on how to do this, please refer to:

- Supported [Corruption types](./docs/corruption_types.md)
- [Configuration](./docs/configuration.md) parameters for the corruptions

## 📑 Supported Datasets

Currently, **RobuSER** supports the:

- [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) dataset.

> More datasets will be added soon.

## 📈 Usage

**RobuSER** provides two main approaches for applying audio corruptions:

### 🎯 Method 1: Batch Dataset Corruption (`corrupt_dataset.py`)

This method applies the same corruption types and levels to all audio files in a dataset, based on a YAML configuration file.

1. Modify the `config.yml` to specify the corruption types and levels.
2. Then you can run the `corrupt_dataset.py` script

```
usage: corrupt_dataset.py [-h] -i INPUT -o OUTPUT [-f] [-s] [-d DATASET] [-c CONFIG]

Corrupt the dataset

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path of the original dataset
  -o OUTPUT, --output OUTPUT
                        Path where the corrupted versions of the dataset will be saved
  -f, --force           Force overwrite the corrupted dataset if it already exists
  -s, --skip_copy       Skip copying the original dataset to the corrupted dataset path and only generate the corrupted wav files.
  -d DATASET, --dataset DATASET
                        Name of the dataset (e.g. iemocap)
  -c CONFIG, --config CONFIG
                        Path to the YAML configuration for the corruptions
```

Example for IEMOCAP:

```
python3 -m robuser.dataset_corruption.corrupt_dataset -i <dataset_path> -o <output_path> -d iemocap
```

🚨 You can use the script to corrupt any directory (without any labels), by not providing a specific dataset with
the `-d` flag. Example:

```
python3 -m robuser.dataset_corruption.corrupt_dataset -i <dataset_path> -o <output_path> --skip_copy
```

The corrupted datasets will be saved in the specified output path.
The `robuser_config.yaml` file, with the corruption configuration, will be generated in the
corrupted dataset's root. Additionally, for certain types of corruptions, the `robuser_metadata.csv` file will also be
created.
This CSV file contains information about the applied corruptions for each original utterance, including, for instance,
the specific noise file
used for background noise corruption or the impulse response file used for impulse response corruption.


### 🎯 Method 2: Per-File Custom Corruption (`corrupt_dataset_per_file.py`)

This method allows you to apply **different corruption types and parameters to individual audio files** based on a CSV specification.

```
usage: corrupt_dataset_per_file.py [-h] -i INPUT [-f]

Apply audio corruptions based on CSV specifications

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the CSV file containing corruption specifications
  -f, --force           Force overwrite output files if they already exist
```

#### CSV Format

The CSV file must contain the following columns:
- `audio_file_path`: Path to the input audio file
- `corruption_type`: Type of corruption (gaussian, compression, clipping_distortion, gain_transition, impulse_response, content)
- `corruption_metadata`: JSON string with corruption parameters
- `output_file_path`: Path where the corrupted audio will be saved

#### Example CSV Content

```csv
audio_file_path,corruption_type,corruption_metadata,output_file_path
example.wav,content,"{""content_dataset_path"": ""datasets/ESC-50"", ""snr"": 10}",output/content_snr10.wav
example.wav,gaussian,"{""snr"": 10}",output/gaussian_snr10.wav
example.wav,gain_transition,"{""min_max_gain_db"": [-20.0,-10.0]}",output/gain_transition.wav
```

#### Corruption Metadata Examples

- **Gaussian noise**: `{"snr": 10}`
- **Compression**: `{"bit_rate": 16}`
- **Clipping distortion**: `{"max_percentile_threshold": 20}`
- **Gain transition**: `{"min_max_gain_db": [-40.0, -20.0]}`
- **Impulse response**: `{"ir_path": "/path/to/impulse/responses", "rt60_range": [0.1, 0.5]}`
- **Content noise**: `{"content_dataset_path": "/path/to/noise/dataset", "snr": 10}`

#### Usage Example

```
python3 -m robuser.dataset_corruption.corrupt_dataset_per_file -i examples/example_corrupt_dataset_per_file.csv
```

You can also download the [examples.html](examples/examples.html) file, to listen to corrupted versions of 4
different (neutral, happy, sad, and angry) utterances.


## 📊 Evaluating the model predictions

For the supported annotated datasets, we also provide scripts to evaluate your model
predictions.
After you train your model, export the predictions to a CSV file with the following format:

```
filename,prediction
Ses01F_impro01_F000.wav,neutral
...
```

Then you can run the `evaluate.py` script:

```
usage: evaluate.py [-h] -csv PREDICTIONS -p DATA_PATH -d {iemocap}

Evaluate the model on the test set

optional arguments:
  -h, --help            show this help message and exit
  -csv PREDICTIONS, --predictions PREDICTIONS
                        Path to the predictions CSV file
  -p DATA_PATH, --data_path DATA_PATH
                        Path to the dataset
  -d {iemocap}, --dataset {iemocap}
                        Name of the dataset
```

Example for IEMOCAP:

```
python3 robuser.evaluation.evaluate -csv <predictions_path> -p <dataset_path> -d iemocap
```

## 📈 Robustness Evaluation

After you've evaluated your model on the corrupted datasets, you can calculate the Corruption Error (CE) and Relative
Corruption Error metrics.
Fill the `results/model_metrics.json` file with the error rate of your model on the clean and corrupted datasets, and then run
the `calculate_ce.py` script:

```
python3 robuser.evaluation.calculate_ce -b <baseline_metrics.json> -i results/model_metrics.json
```

Example for IEMOCAP:

```
python3 robuser.evaluation.calculate_ce -b results/iemocap_baseline_metrics.json -i results/model_metrics.json
```

The script will output the CE and relative CE metrics as defined in the section _Robustness evaluation_ of the paper.

## 📝 How to contribute

If you want to add support for a new dataset, please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file.
Note that although this project is focused on SER, it can be used for any speech dataset/task such as speech
recognition, speaker recognition, etc.
