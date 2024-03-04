# 💪 ROBUSER
A robustness evaluation benchmarking procedure for Speech Emotion Recognition (SER) 💬


## 💁 Installation guidelines


- Install & activate poetry *(used for managing dependencies)*

```
pip3 install poetry
poetry shell
poetry install --no-root
```

- Install FFmpeg & libasound2-dev:

```
sudo apt install ffmpeg libasound2-dev
```

> The dependencies have been installed 👏


# 🚀 Code Structure

- `src/parsing`: contains the code for parsing the datasets.
- `src/evaluation`: contains the code for evaluating the model predictions.
- `src/noise_generation`: contains the scripts for adding

## 📰 Documentation

For detailed documentation, please refer to:
-  Supported [Corruption types](./docs/corruption_types.md)
- [Configuration](./docs/configuration.md) parameters for the corruptions

## 📑 Supported Datasets

Currently, **ROBUSER** supports the `IEMOCAP` dataset – [Download dataset](https://sail.usc.edu/iemocap/iemocap_release.htm)


## 📈 Usage

1. Modify the `config.yaml` to specify the corruption types and levels.
2. Then you can run the `corrupt_dataset.py` script in the `src/noise_generation` directory. 

```
usage: corrupt_dataset.py [-h] -i INPUT -o OUTPUT [-f] [-s] -d DATASET [-c CONFIG]

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

- Example for IEMOCAP:

```
python3 src/noise_generation/corrupt_dataset.py -i <dataset_path> -o <output_path> -d iemocap
```

You can also find multiple corruption [examples](./docs/examples.md) of the same audio file.