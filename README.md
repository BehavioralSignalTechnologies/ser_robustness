# 💪 ROBUSER [TODO]
A robustness evaluation benchmarking procedure for Speech Emotion Recognition (SER) 💬


## 📑 Input dataset structure 

[TODO]

## 💁 Installation guidelines

To get started with ROBUSER, follow these simple steps:

- Install & activate poetry (used for managing dependencies)

```
pip3 install poetry
poetry shell
poetry install --no-root
```

- Install FFmpeg (for audio processing):

```
sudo apt install ffmpeg libasound2-dev
```

> The dependencies have been installed 👏


## 📈 Usage

Edit the `config.yaml` file to specify the corruption types and levels you want to evaluate. Then, run the following command:

```
python3 src/noise_generation/corrupt_dataset.py -i <dataset_path> -o <output_path> -d <dataset_name>
```

# 🚀 Code Structure

- `src/parsing`: contains the code for parsing the datasets.
- `src/evaluation`: contains the code for evaluating the model predictions.
- `src/noise_generation`: contains the scripts for adding

## 📰 Documentation

For detailed documentation, please refer to:
-  [Corruption types](./docs/corruption_types.md) file, which outlines the supported types of corruptions.
