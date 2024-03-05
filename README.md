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


## 📰 Documentation

Before running the dataset corruption scripts you first need to download the 
required noise datasets and configure the respective parameters. For detailed 
documentation on how to do this, please refer to:
-  Supported [Corruption types](./docs/corruption_types.md)
- [Configuration](./docs/configuration.md) parameters for the corruptions

## 📑 Supported Datasets

Currently, **ROBUSER** supports the:
- [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) dataset.

> More datasets will be added soon.


## 📈 Usage

1. Modify the `config.yaml` to specify the corruption types and levels.
2. Then you can run the `corrupt_dataset.py` script in the `src/noise_generation` directory. 

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

- Example for IEMOCAP:

```
python3 src/noise_generation/corrupt_dataset.py -i <dataset_path> -o <output_path> -d iemocap
```

We have also created the [examples.ipynb](src/noise_generation/examples.ipynb) notebook, in which we have used the different types of corruptions provided in this notebook, to produce the corrupted versions of the same audio file. To listen to each corrupted audio, you can download the [examples.html](src/noise_generation/examples.html) file.

🚨 You can use the script to corrupt any directory (without any labels), 
by not providing a specific dataset with the `-d` flag.

- Example:

```
python3 src/noise_generation/corrupt_dataset.py -i <dataset_path> -o <output_path> --skip_copy
```

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
python3 src/evaluation/evaluate.py -csv <predictions_path> -p <dataset_path> -d iemocap
```

## 📝 How to contribute

If you want to add support for a new dataset, please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file.
Note that although this project is focused on SER, it can be used for any speech dataset/task such as speech recognition, speaker recognition, etc.
