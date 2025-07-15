"""
Applies corruptions to an audio dataset based on a CSV file, which specifies the corruption type and metadata per audio file.
"""

import argparse
import csv
import json
import os

import librosa
import soundfile as sf
from tqdm import tqdm
from frozendict import frozendict


from robuser.corruptions.get_corruption import get_corruption


def parse_corruption_metadata(metadata_str):
    """
    Parse the corruption metadata from string format to dictionary.

    Args:
        metadata_str (str): JSON string containing corruption parameters

    Returns:
        dict: Parsed corruption configuration
    """
    try:
        metadata_dict = json.loads(metadata_str)
        # Sort by keys to ensure consistent ordering
        metadata_dict = {k: metadata_dict[k] for k in sorted(metadata_dict.keys())}
        return frozendict(metadata_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in corruption metadata: {metadata_str}. Error: {e}")


def apply_corruption_from_csv(csv_file_path, force=False):
    """
    Apply corruptions to audio files based on specifications in a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing corruption specifications
        force (bool): Force overwrite output files if they already exist
    """

    corruptions_to_apply = {}
    corruption_types = set()

    # Read the CSV file containing per-file corruption specifications
    with open(csv_file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        # Validate CSV headers
        required_headers = [
            "audio_file_path",
            "corruption_type",
            "corruption_metadata",
            "output_file_path",
        ]
        if not all(header in reader.fieldnames for header in required_headers):
            raise ValueError(
                f"CSV file must contain headers: {required_headers}. Found: {reader.fieldnames}"
            )

        for row in reader:
            corr_metadata = parse_corruption_metadata(row["corruption_metadata"])
            corruption_key = (row["corruption_type"], corr_metadata)
            if corruption_key not in corruptions_to_apply:
                corruptions_to_apply[corruption_key] = []
            corruptions_to_apply[corruption_key].append(
                (row["audio_file_path"], row["output_file_path"])
            )
            corruption_types.add(row["corruption_type"])

    # Print the number of audio files for each corruption type
    for corruption_type in corruption_types:
        len_audio_files = 0
        len_corruptions = 0
        for corruption_key, audio_files in corruptions_to_apply.items():
            if corruption_key[0] == corruption_type:
                len_audio_files += len(audio_files)
                len_corruptions += 1
        print(f"Number of audio files for {corruption_type}: {len_audio_files}, {len_corruptions} unique corruption configurations")

    # Apply the corruptions
    for (corruption_type, corruption_metadata), audio_files in tqdm(
        corruptions_to_apply.items(), desc="Applying corruptions"
    ):
        corruption_class = get_corruption(corruption_type)
        corruption = corruption_class(corruption_metadata)

        for audio_file_path, output_file_path in audio_files:
            # Check if output file already exists
            if os.path.exists(output_file_path) and not force:
                print(
                    f"Warning: Output file already exists: {output_file_path}. Use --force to overwrite. Skipping."
                )
                continue
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # Load the audio file
            audio, sr = librosa.load(audio_file_path, sr=None)

            # Apply the corruption
            augmented_audio, corruption_type = corruption.run(audio, sr)
            sf.write(output_file_path, augmented_audio, sr)


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Apply audio corruptions based on CSV specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV Format:
The CSV file must contain the following columns:
- audio_file_path: Path to the input audio file
- corruption_type: Type of corruption (gaussian, compression, clipping_distortion, gain_transition, impulse_response, content)
- corruption_metadata: JSON string with corruption parameters
- output_file_path: Path where the corrupted audio will be saved

Example corruption_metadata for different types:
- gaussian: {"snr": 10}
- compression: {"bit_rate": 16}
- clipping_distortion: {"max_percentile_threshold": 20}
- gain_transition: {"min_max_gain_db": [-40.0, -20.0]}
- impulse_response: {"ir_path": "/path/to/impulse/responses", "rt60_range": [0.1, 0.5]}
- content: {"content_dataset_path": "/path/to/noise/dataset", "snr": 10}
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the CSV file containing corruption specifications",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite output files if they already exist",
    )
    return parser.parse_args()


def main():
    """Main entry point for the console script"""
    args = parse_arguments()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"CSV file not found: {args.input}")

    apply_corruption_from_csv(args.input, args.force)
    print("Corruption application completed!")


if __name__ == "__main__":
    main()
