# 💪 ROBUSER
A robustness evaluation benchmarking procedure for Speech Emotion Recognition (SER) 💬


## 💁 Installation guidelines


- Install & activate poetry (used for managing dependencies)

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


## 📈 Usage

Modify the `config.yaml` file to enable/disable the corruption types and specify the corruption levels you want to evaluate. Then you can run the `corruption_dataset.py` script. 
- Example:

```
python3 src/noise_generation/corrupt_dataset.py -i <dataset_path> -o <output_path> -d <dataset_name>
```

>, where `dataset_path` is the path to the original dataset, `output_path` is the path to the corrupted dataset and `dataset_name` is the type of dataset (e.g. `iemocap`)


# 🚀 Code Structure

- `src/parsing`: contains the code for parsing the datasets.
- `src/evaluation`: contains the code for evaluating the model predictions.
- `src/noise_generation`: contains the scripts for adding

## 📰 Documentation

For detailed documentation, please refer to:
-  [Corruption types](./docs/corruption_types.md), which outlines the supported types of corruptions.
- [Configuration](.docs/configuration.md), which analyzes the configuration parameters.

## 📑 Supported Datasets

Currently, only the `IEMOCAP` dataset is supported. [Download dataset](https://sail.usc.edu/iemocap/iemocap_release.htm)