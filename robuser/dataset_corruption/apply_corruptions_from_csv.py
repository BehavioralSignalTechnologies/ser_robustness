import argparse
import csv
import json
import os

import librosa
import soundfile as sf
from tqdm import tqdm

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
        return json.loads(metadata_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in corruption metadata: {metadata_str}. Error: {e}")


def apply_corruption_from_csv(csv_file_path, force=False):
    """
    Apply corruptions to audio files based on specifications in a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing corruption specifications
        force (bool): Force overwrite output files if they already exist
    """

    # Read the CSV file
    corruptions_to_apply = []

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
            corruptions_to_apply.append(
                {
                    "audio_file_path": row["audio_file_path"],
                    "corruption_type": row["corruption_type"],
                    "corruption_metadata": parse_corruption_metadata(row["corruption_metadata"]),
                    "output_file_path": row["output_file_path"],
                }
            )

    print(f"Found {len(corruptions_to_apply)} corruptions to apply")

    # Process each corruption
    for corruption_spec in tqdm(corruptions_to_apply, desc="Applying corruptions"):
        audio_file_path = corruption_spec["audio_file_path"]
        corruption_type = corruption_spec["corruption_type"]
        corruption_config = corruption_spec["corruption_metadata"]
        output_file_path = corruption_spec["output_file_path"]

        # Check if input file exists
        if not os.path.exists(audio_file_path):
            print(f"Warning: Input file does not exist: {audio_file_path}. Skipping.")
            continue

        # Check if output file already exists
        if os.path.exists(output_file_path) and not force:
            print(
                f"Warning: Output file already exists: {output_file_path}. Use --force to overwrite. Skipping."
            )
            continue

        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # Get the corruption class
            corruption_class = get_corruption(corruption_type)

            # Initialize the corruption
            corruption = corruption_class(corruption_config)

            # Load the audio file
            audio, sr = librosa.load(audio_file_path, sr=None)

            # Apply the corruption
            augmented_audio, corruption_type = corruption.run(audio, sr)
            sf.write(output_file_path, augmented_audio, sr)

            print(
                f"Successfully applied {corruption_type} corruption to {audio_file_path} -> {output_file_path}"
            )

        except Exception as e:
            print(f"Error applying {corruption_type} corruption to {audio_file_path}: {e}")
            # Clean up partial output file if it was created
            if os.path.exists(output_file_path):
                os.remove(output_file_path)


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
