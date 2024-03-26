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

- Supported [Corruption types](./docs/corruption_types.md)
- [Configuration](./docs/configuration.md) parameters for the corruptions

## 📑 Supported Datasets

Currently, **RobuSER** supports the:

- [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) dataset.

> More datasets will be added soon.

## 📈 Usage

1. Modify the `config.yml` to specify the corruption types and levels.
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

Example for IEMOCAP:

```
python3 src/noise_generation/corrupt_dataset.py -i <dataset_path> -o <output_path> -d iemocap
```

You can also download the [examples.html](src/noise_generation/examples.html) file, to listen to corrupted versions of 4
different (neutral, happy, sad, and angry) utterances.

🚨 You can use the script to corrupt any directory (without any labels), by not providing a specific dataset with
the `-d` flag. Example:

```
python3 src/noise_generation/corrupt_dataset.py -i <dataset_path> -o <output_path> --skip_copy
```

The corrupted datasets will be saved in the specified output path.
The `robuser_config.yaml` file, with the corruption configuration, will be generated in the
corrupted dataset's root. Additionally, for certain types of corruptions, the `robuser_metadata.csv` file will also be
created.
This CSV file contains information about the applied corruptions for each original utterance, including, for instance,
the specific noise file
used for background noise corruption or the impulse response file used for impulse response corruption.

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

## 📈 Robustness Evaluation

After you've evaluated your model on the corrupted datasets, you can calculate the Corruption Error (CE) and Relative
Corruption Error metrics.
Fill the `results/model_metrics.json` file with the error rates of your model on the clean and corrupted datasets, and then run
the `calculate_ce.py` script:

```
python3 src/evaluation/calculate_ce.py -b <baseline_metrics.json> -i results/model_metrics.json
```

Example for IEMOCAP:

```
python3 src/evaluation/calculate_ce.py -b results/iemocap_baseline_metrics.json -i results/model_metrics.json
```

The script will output the CE and relative CE metrics as defined in the section _Robustness evaluation_ of the paper.

## 📝 How to contribute

If you want to add support for a new dataset, please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file.
Note that although this project is focused on SER, it can be used for any speech dataset/task such as speech
recognition, speaker recognition, etc.
