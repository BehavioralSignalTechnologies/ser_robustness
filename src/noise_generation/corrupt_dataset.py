import argparse
import json
import os
import shutil
import sys

import librosa
from scipy.io import wavfile
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from parsing.get_parser import get_parser_for_dataset
from noise_generation.get_corruption import get_corruption


def copy_dataset(original_dataset_path, corrupted_dataset_path, ignore_extensions=[".wav"]):
    """
    Copies the original dataset to the corrupted dataset path, ignoring files with the specified extensions.

    Args:
        original_dataset_path (str): path to the original dataset
        corrupted_dataset_path (str): path to the corrupted dataset
        ignore_extensions (list): list of file extensions to ignore
    """
    for root, dirs, files in os.walk(original_dataset_path):
        for file in files:
            if not any([file.endswith(extension) for extension in ignore_extensions]):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, original_dataset_path)
                output_file_path = os.path.join(corrupted_dataset_path, relative_path)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                shutil.copy2(file_path, output_file_path)


def corrupt(original_dataset_path, corrupted_dataset_path, dataset_name, corruption_type, corruption_config,
            force=False):
    """
    Corrupts the original dataset with the specified corruption type and configuration.

    Args:
        original_dataset_path (str): path to the original dataset
        corrupted_dataset_path (str): path to the corrupted dataset
        dataset_name (str): name of the dataset (e.g. iemocap)
        corruption_type (str): type of corruption (e.g. content)
        corruption_config (dict): configuration for the corruption
        force (bool): force overwrite the corrupted dataset if it already exists
    """

    # Parse the original dataset
    parser_class = get_parser_for_dataset(dataset_name)
    parser = parser_class(original_dataset_path)
    annotated_files_dict = parser.run_parser()

    # Copy the original dataset to the corrupted dataset path
    # This is a convenient dataset-agnostic way to keep the original dataset structure and metadata
    if os.path.exists(corrupted_dataset_path):
        if force:
            shutil.rmtree(corrupted_dataset_path)
        else:
            raise FileExistsError(
                f"The corrupted dataset already exists at {corrupted_dataset_path}. Use --force to overwrite it.")

    # Copy ignoring .wav files
    copy_dataset(original_dataset_path, corrupted_dataset_path, ignore_extensions=[".wav"])

    print(f"Original dataset copied to {corrupted_dataset_path}")

    # Corrupt the dataset
    corruption_class = get_corruption(corruption_type)
    corruption = corruption_class(corruption_config)

    # Metadata for the corrupted dataset
    robuser_metadata = {}

    for file_path in tqdm(annotated_files_dict, desc="Corrupting dataset"):
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)
        augmented_audio, noise_type = corruption.run(audio, sr)

        # file_path is an absolute path, find the relative path to the original_dataset_path
        relative_path = os.path.relpath(file_path, original_dataset_path)
        output_file_path = os.path.join(corrupted_dataset_path, relative_path)

        # Save the corrupted audio file
        robuser_metadata[output_file_path] = noise_type
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        wavfile.write(output_file_path, sr, augmented_audio)

    # Save the metadata
    if not all(value is None for value in robuser_metadata.values()):
        metadata_path = os.path.join(corrupted_dataset_path, "robuser_metadata.csv")

        # Save as CSV
        with open(metadata_path, "w") as file:
            file.write("file_path,noise_type\n")
            for key, value in robuser_metadata.items():
                file.write(f"{key},{value}\n")

        print(f"Metadata saved to {metadata_path}")


def parse_arguments():
    """!
    @brief Parse Arguments for corrupting the dataset.
    """
    args_parser = argparse.ArgumentParser(description="Corrupt the dataset")
    args_parser.add_argument('-i', '--input', required=True,
                             help="Path of the original dataset")
    args_parser.add_argument('-o', '--output', required=True,
                             help="Path of the corrupted dataset")
    args_parser.add_argument('-f', '--force', action="store_true",
                             help="Force overwrite the corrupted dataset if it already exists")
    args_parser.add_argument('-d', '--dataset', required=True,
                             help="Name of the dataset (e.g. iemocap)")
    args_parser.add_argument('-t', '--type', required=True,
                             help="Type of corruption (e.g. content)")
    args_parser.add_argument('-c', '--config', required=True, type=json.loads,
                             help="Configuration for the corruption")
    return args_parser.parse_args()


if __name__ == '__main__':
    # command line example:
    # python3 corrupt_dataset.py -i /data_drive/iemocap -o /data_drive/iemocap_corrupted -d iemocap -t content -c '{"content_dataset_path": "/data_drive/ESC-50-master", "snr": 0}'

    args = parse_arguments()

    corrupt(args.input, args.output, args.dataset, args.type, args.config, args.force)
