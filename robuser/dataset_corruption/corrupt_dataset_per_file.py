"""
Applies corruptions to an audio dataset based on a CSV file, which specifies the corruption type and metadata per audio file.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import soundfile as sf
from tqdm import tqdm
from frozendict import frozendict


from robuser.corruptions.get_corruption import get_corruption


def _freeze(value):
    """Make a parsed JSON value hashable: dicts -> frozendict, lists -> tuples (recursively)."""
    if isinstance(value, dict):
        return frozendict({k: _freeze(value[k]) for k in sorted(value)})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def parse_corruption_metadata(metadata_str):
    """
    Parse the corruption metadata from string format to dictionary.

    Args:
        metadata_str (str): JSON string containing corruption parameters

    Returns:
        dict: Parsed corruption configuration (hashable: nested dicts and lists, e.g. chain steps, are frozen too)
    """
    try:
        metadata_dict = json.loads(metadata_str)
        return _freeze(metadata_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in corruption metadata: {metadata_str}. Error: {e}")


def _apply_group(args):
    """Apply one corruption configuration to its files. Returns (applied_noise_paths, info_rows)."""
    corruption_type, corruption_metadata, audio_files, force = args
    corruption = get_corruption(corruption_type)(corruption_metadata)
    applied_noise_paths, info_rows = {}, []
    for audio_file_path, output_file_path in audio_files:
        # Check if output file already exists
        if os.path.exists(output_file_path) and not force:
            print(f"Warning: Output file already exists: {output_file_path}. Use --force to overwrite. Skipping.")
            continue
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Apply the corruption
        try:
            # Load the audio file
            audio, sr = librosa.load(audio_file_path, sr=None)
            augmented_audio, info = corruption.run_with_info(audio, sr)
            sf.write(output_file_path, augmented_audio, sr)
            applied_noise_paths[output_file_path] = info.get("applied_noise_path", info.get("noise_path"))
            info_rows.append({"audio_file_path": audio_file_path, "output_file_path": output_file_path,
                              "corruption_type": corruption_type, **info})
        except Exception as e:
            print(f"Error applying corruption to {audio_file_path}: {e}. Skipping this file.")
            continue
    return applied_noise_paths, info_rows


def apply_corruption_from_csv_with_info(csv_file_path, force=False, workers=1):
    """
    Apply corruptions to audio files based on specifications in a CSV file, and return what was applied.

    Args:
        csv_file_path (str): Path to the CSV file containing corruption specifications
        force (bool): Force overwrite output files if they already exist
        workers (int): Number of parallel processes (1 = sequential)
    Returns:
        tuple: (applied noise paths per output file, list of per-file info rows: the labels and parameters returned by
            each corruption's run_with_info)
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

    applied_noise_paths = {}
    # Print the number of audio files for each corruption type
    for corruption_type in corruption_types:
        len_audio_files = 0
        len_corruptions = 0
        for corruption_key, audio_files in corruptions_to_apply.items():
            if corruption_key[0] == corruption_type:
                len_audio_files += len(audio_files)
                len_corruptions += 1
        print(f"Number of audio files for {corruption_type}: {len_audio_files}, {len_corruptions} unique corruption configurations")

    # Apply the corruptions (one task per unique configuration; workers > 1 runs them in parallel)
    tasks = [(ctype, meta, files, force) for (ctype, meta), files in corruptions_to_apply.items()]
    info_rows = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(tqdm(ex.map(_apply_group, tasks, chunksize=16), total=len(tasks), desc="Applying corruptions"))
    else:
        results = [_apply_group(t) for t in tqdm(tasks, desc="Applying corruptions")]
    for paths, rows in results:
        applied_noise_paths.update(paths)
        info_rows.extend(rows)

    return applied_noise_paths, info_rows


def apply_corruption_from_csv(csv_file_path, force=False):
    """
    Apply corruptions to audio files based on specifications in a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing corruption specifications
        force (bool): Force overwrite output files if they already exist
    Returns:
        dict: Dictionary mapping input audio file paths to the paths of applied noise files (for applicable corruptions)
    """
    applied_noise_paths, _ = apply_corruption_from_csv_with_info(csv_file_path, force)
    return applied_noise_paths


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
    parser.add_argument(
        "--info_output",
        default="corruption_info.csv",
        help="CSV with the labels and parameters applied to each output file",
    )
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of parallel processes")
    return parser.parse_args()


def main():
    """Main entry point for the console script"""
    args = parse_arguments()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"CSV file not found: {args.input}")

    applied_noise_paths, info_rows = apply_corruption_from_csv_with_info(args.input, args.force, args.workers)
    output_file = "applied_noise_paths.csv"
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_file_path", "applied_noise_path"])
        for audio_file_path, applied_noise_path in applied_noise_paths.items():
            writer.writerow([audio_file_path, applied_noise_path])
    print(f"CSV with the applied noise paths saved to {output_file}")
    fields = list(dict.fromkeys(k for row in info_rows for k in row))
    with open(args.info_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(info_rows)
    print(f"CSV with the applied labels and parameters saved to {args.info_output}")
    print("Corruption application completed!")


if __name__ == "__main__":
    main()
