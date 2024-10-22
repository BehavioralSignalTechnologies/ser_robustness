import argparse
import itertools
import os
import shutil
import sys

import librosa
import yaml
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import resample_dataset
from parsing.get_parser import get_parser_for_dataset
from noise_generation.get_corruption import get_corruption


def copy_dataset(original_dataset_path, corrupted_dataset_path, ignore_extensions=None):
    """
    Copies the original dataset to the corrupted dataset path, ignoring files with the specified extensions.

    Args:
        original_dataset_path (str): path to the original dataset
        corrupted_dataset_path (str): path to the corrupted dataset
        ignore_extensions (list): list of file extensions to ignore
    """
    if ignore_extensions is None:
        ignore_extensions = [".wav"]

    print(f"Copying the original dataset to: {corrupted_dataset_path}")
    for root, _, files in os.walk(original_dataset_path):
        for file in files:
            if not any([file.endswith(extension) for extension in ignore_extensions]):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, original_dataset_path)
                output_file_path = os.path.join(corrupted_dataset_path, relative_path)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                shutil.copy2(file_path, output_file_path)


def corrupt_dataset(original_dataset_path, corrupted_dataset_path, dataset_name, corruption_type, corruption_config,
                    force=False, skip_copy=False):
    """
    Corrupts the original dataset with the specified corruption type and configuration.

    Args:
        original_dataset_path (str): path to the original dataset
        corrupted_dataset_path (str): path to the corrupted dataset
        dataset_name (str): name of the dataset (e.g. iemocap)
        corruption_type (str): type of corruption (e.g. content)
        corruption_config (dict): configuration for the corruption
        force (bool): force overwrite the corrupted dataset if it already exists
        skip_copy (bool): skip copying the original dataset to the corrupted dataset path
    """

    # Parse the original dataset
    if dataset_name != None:
        parser_class = get_parser_for_dataset(dataset_name)
        parser = parser_class(original_dataset_path)
        files_dict = parser.run_parser()
        files_dict = dict(sorted(files_dict.items()))
    else: 
        files_dict = {}
        for root, _, files in os.walk(original_dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    files_dict[file_path] = None

    # Check if the corrupted dataset already exists
    if os.path.exists(corrupted_dataset_path):
        if force:
            shutil.rmtree(corrupted_dataset_path)
        else:
            raise FileExistsError(
                f"The corrupted dataset already exists at {corrupted_dataset_path}. Use --force to overwrite it.")

    # Copy the original dataset to the corrupted dataset path
    # This is a convenient dataset-agnostic way to keep the original dataset structure and metadata
    if not skip_copy:
        copy_dataset(original_dataset_path, corrupted_dataset_path, ignore_extensions=[".wav"])

    # Initialize the corruption class
    corruption_class = get_corruption(corruption_type)

    if corruption_type == "impulse_response":
        # Resample the reverberation dataset if needed
        resampled_path = corruption_config['ir_path'] + "_resampled"
        shutil.copytree(corruption_config['ir_path'], resampled_path, dirs_exist_ok=True)
        resample_dataset(original_dataset_path, resampled_path)
        corruption_config['ir_path'] = resampled_path

    corruption = corruption_class(corruption_config)

    # Metadata for the corrupted dataset
    robuser_metadata = {}

    # Corrupt the dataset
    for file_path in tqdm(files_dict, desc=f"Corrupting dataset with '{corruption_type}' corruption"):
        if corruption_type == 'compression':
            _, sr = librosa.load(file_path, sr=None)

            relative_path = os.path.relpath(file_path, original_dataset_path)
            output_file_path = os.path.join(corrupted_dataset_path, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            corruption.run(file_path, sr, output_file_path)
            robuser_metadata[output_file_path] = None
        else:
            # Load the audio file
            audio, sr = librosa.load(file_path, sr=None)
            augmented_audio, noise_type = corruption.run(audio, sr)

            # file_path is an absolute path, find the relative path to the original_dataset_path
            relative_path = os.path.relpath(file_path, original_dataset_path)
            output_file_path = os.path.join(corrupted_dataset_path, relative_path)

            # Save the corrupted audio file
            robuser_metadata[output_file_path] = noise_type
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            sf.write(output_file_path, augmented_audio, sr)

    # Save the metadata
    if not all(value is None for value in robuser_metadata.values()):
        metadata_path = os.path.join(corrupted_dataset_path, "robuser_metadata.csv")

        # Save as CSV
        with open(metadata_path, "w") as file:
            file.write("file_path,noise_type\n")
            for key, value in robuser_metadata.items():
                file.write(f"{key},{value}\n")

        print(f"Metadata saved to {metadata_path}")


def parse_config(config):
    """
    Parses the configuration for the corruptions.

    Args:
        config (dict): configuration for the corruptions

    Returns:
        list of tuples: list of tuples with the corruption type and configuration
    """
    corruptions = []

    for corruption_type, corruption_config in config.items():
        if not corruption_config.pop("enabled", False):
            continue

        for values in itertools.product(*corruption_config.values()):
            corruptions.append([corruption_type, dict(zip(corruption_config, values))])

    print(f"Will apply the following {len(corruptions)} corruptions:")
    for corruption in corruptions:
        print(corruption)

    return corruptions


def get_corruption_str(corruption_type, corruption_config):
    """
    Returns a string representation of the corruption type and configuration.
    """
    config_str = ""
    for key, value in corruption_config.items():
        if key == "enabled":
            continue
        if key == "content_dataset_path" or key == "ir_path":
            config_str += f"_dataset_{os.path.basename(value)}"
        elif isinstance(value, list):
            config_str += f"_{key}_{'_'.join(map(str, value))}"
        else:
            config_str += f"_{key}_{value}"

    return corruption_type + config_str


def corrupt(dataset_name, original_dataset_path, corrupted_datasets_path, corruptions_config, force=False, skip_copy=False):
    """
    Corrupts the original dataset with the specified corruption type and configuration.

    Args:
        dataset_name (str): name of the dataset (e.g. iemocap)
        original_dataset_path (str): path to the original dataset
        corrupted_datasets_path (str): path to the corrupted datasets
        corruptions_config (dict): configuration for the corruption
        force (bool): force overwrite the corrupted dataset if it already exists
        skip_copy (bool): skip copying the original dataset to the corrupted dataset path
    """

    corruptions_list = parse_config(corruptions_config)
    for corruption_type, corruption_config in tqdm(corruptions_list, desc="Corrupting datasets"):
        if dataset_name is None:
            corrupted_dataset_path = os.path.join(corrupted_datasets_path,
                                              f"{get_corruption_str(corruption_type, corruption_config)}")
        else:
            corrupted_dataset_path = os.path.join(corrupted_datasets_path,
                                              f"{dataset_name}_{get_corruption_str(corruption_type, corruption_config)}")
        try:
            corrupt_dataset(original_dataset_path, corrupted_dataset_path, dataset_name, corruption_type,
                            corruption_config, force, skip_copy)
            with open(os.path.join(corrupted_dataset_path, "robuser_config.yaml"), "w") as file_:
                yaml.dump(corruption_config, file_)
        except Exception as e:
            print(f"Error while corrupting the dataset with '{corruption_type}' corruption: {e}")
            shutil.rmtree(corrupted_dataset_path, ignore_errors=True)


def parse_arguments():
    """!
    @brief Parse Arguments for corrupting the dataset.
    """
    args_parser = argparse.ArgumentParser(description="Corrupt the dataset")
    args_parser.add_argument('-i', '--input', required=True,
                             help="Path of the original dataset")
    args_parser.add_argument('-o', '--output', required=True,
                             help="Path where the corrupted versions of the dataset will be saved")
    args_parser.add_argument('-f', '--force', action="store_true",
                             help="Force overwrite the corrupted dataset if it already exists")
    args_parser.add_argument('-s', '--skip_copy', action="store_true",
                             help="Skip copying the original dataset to the corrupted dataset path and only generate the corrupted wav files.")
    args_parser.add_argument('-d', '--dataset', required=False,
                             help="Name of the dataset (e.g. iemocap)")
    args_parser.add_argument('-c', '--config', type=str, default="config.yml",
                             help="Path to the YAML configuration for the corruptions")
    return args_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    corrupt(args.dataset, args.input, args.output, config, args.force, args.skip_copy)
