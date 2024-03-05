import os
import sys
import argparse
from sklearn.metrics import accuracy_score, recall_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from parsing.get_parser import get_parser_for_dataset


def parse_csv(preds_csv):
    """
    Parse the predictions CSV file into a dictionary
    Args:
        preds_csv: Path to the predictions CSV file with the following columns:
                   file_name (e.g. Ses05F_impro02_M007.wav), emotion
    Returns:
        Dictionary of {file_name: {"emotion": emotion}}
    """
    preds = {}
    with open(preds_csv, "r") as file:
        for line in file:
            file_name, emotion = line.strip().split(",")
            preds[file_name] = {"emotion": emotion}

    return preds


def evaluate_iemocap(preds, targets):
    """
    Evaluate the model on IEMOCAP by performing 10-fold cross-validation
    Args:
        preds: Predictions from the model as dictionary of {file_name: {"emotion": emotion}}
        targets: True labels: dictionary of {file_name: {"emotion": emotion, "fold": fold, "speaker_id": speaker_id}}
    Returns:
        Dictionary of {fold: [weighted_accuracy, unweighted_accuracy]}
        List of [average_weighted_accuracy, average_unweighted_accuracy]
    """

    # Only keep the expected classes
    expected_classes = {"neutral", "happy", "sad", "angry"}
    preds = {key: value for key, value in preds.items() if value["emotion"] in expected_classes}
    targets = {key: value for key, value in targets.items() if value["emotion"] in expected_classes}

    # Remove prediction keys that are not in the targets
    preds = {key: value for key, value in preds.items() if key in targets}

    # Check that the filenames are the same
    if set(preds.keys()) != set(targets.keys()):
        raise ValueError("Predictions and targets have different keys")

    # Check that the lengths are the same
    if len(preds) != len(targets):
        raise ValueError(f"Predictions and targets have different lengths: {len(preds)} and {len(targets)}")

    # Evaluate with 10-fold cross-validation
    folds = set([target["speaker_id"] for target in targets.values()])
    if len(folds) != 10:
        raise ValueError(f"Expected 10 folds, but got {len(folds)}")

    results = {}
    for fold in sorted(folds):
        fold_filenames = [key for key, value in targets.items() if value["speaker_id"] == fold]

        fold_preds = [preds[key]["emotion"] for key in fold_filenames]
        fold_targets = [targets[key]["emotion"] for key in fold_filenames]

        # Calculate the WA
        wa = accuracy_score(fold_targets, fold_preds)

        # Calculate the UA (macro averaged recall)
        ua = recall_score(fold_targets, fold_preds, average="macro")

        results[fold] = [wa, ua]

    avg_results = [sum([result[0] for result in results.values()]) / len(results),
                   sum([result[1] for result in results.values()]) / len(results)]
    return results, avg_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model on the test set")
    parser.add_argument("-csv", "--predictions", type=str, required=True, help="Path to the predictions CSV file")
    parser.add_argument("-p", "--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("-d", "--dataset", type=str, choices=["iemocap"], required=True, help="Name of the dataset")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    preds = parse_csv(args.predictions)
    parser_class = get_parser_for_dataset(args.dataset)
    print(f"Evaluating the predictions {args.predictions} on the {args.dataset} dataset at {args.data_path}")
    if args.dataset == "iemocap":
        parser = parser_class(args.data_path)
        targets = parser.run_parser()
        targets = {os.path.basename(k): v for k, v in targets.items()}
        results, avg_results = evaluate_iemocap(preds, targets)

        print("-" * 50)
        print("Per-fold results:")
        for fold, (wa, ua) in results.items():
            print(f"Fold {fold}: WA: {wa:.2f}, "
                  f"UA: {ua:.2f}")
        print("-" * 50)
        print(f"Average results: WA: {avg_results[0]:.2f}, "
              f"UA: {avg_results[1]:.2f}")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
