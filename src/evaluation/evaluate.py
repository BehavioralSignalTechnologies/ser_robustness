import argparse
import os
import sys

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix

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


def calc_metrics_from_cm(confusion_matrix):
    metrics = {
        "recall": [],
        "weighted_recall": [],
        "f1_score": [],
    }
    TPs = []
    instances_per_class = []
    for i in range(len(confusion_matrix)):
        TP = confusion_matrix[i][i]
        FN = sum(confusion_matrix[i]) - TP  # Sum the row except for TP to get FN
        FP = sum(confusion_matrix[j][i] for j in range(len(confusion_matrix)) if j != i)

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        # accuracy = TP / sum(confusion_matrix[i]) if sum(confusion_matrix[i]) != 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0.0
        )
        TPs.append(TP)
        instances_per_class.append(sum(confusion_matrix[i]))
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)
    metrics["weighted_accuracy"] = sum(TPs) / sum(instances_per_class)
    metrics["unweighted_accuracy"] = sum(metrics["recall"]) / len(metrics["recall"])
    metrics["macro_f1_score"] = sum(metrics["f1_score"]) / len(metrics["f1_score"])

    return metrics


def evaluate_iemocap(preds, targets):
    """
    Evaluate the model on IEMOCAP by performing 10-fold cross-validation
    Args:
        preds: Predictions from the model as dictionary of {file_name: {"emotion": emotion}}
        targets: True labels: dictionary of {file_name: {"emotion": emotion, "fold": fold, "speaker_id": speaker_id}}
    Returns:
        Dictionary of {fold: [weighted_accuracy, unweighted_accuracy]}
        List of [overal_wa, overall_ua]
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

    results = {}
    for fold in sorted(folds):
        fold_filenames = [key for key, value in targets.items() if value["speaker_id"] == fold]

        fold_preds = [preds[key]["emotion"] for key in fold_filenames]
        fold_targets = [targets[key]["emotion"] for key in fold_filenames]

        # Calculate the WA
        wa = accuracy_score(fold_targets, fold_preds) * 100

        # Calculate the UA (macro averaged recall)
        ua = recall_score(fold_targets, fold_preds, average="macro") * 100

        results[fold] = [wa, ua]

    all_filenames = list(targets.keys())
    all_preds = [preds[key]["emotion"] for key in all_filenames]
    all_targets = [targets[key]["emotion"] for key in all_filenames]

    cm = confusion_matrix(all_targets, all_preds)
    overall_metrics = calc_metrics_from_cm(cm)
    overall_wa = overall_metrics["weighted_accuracy"] * 100
    overall_ua = overall_metrics["unweighted_accuracy"] * 100

    return results, [overall_wa, overall_ua]


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
        results, overall_results = evaluate_iemocap(preds, targets)

        print("-" * 50)
        print("Per-fold results:")
        for fold, (wa, ua) in results.items():
            print(f"Fold {fold}: WA: {wa:.2f}%, "
                  f"UA: {ua:.2f}%")
        print("-" * 50)
        print(f"Overall results: WA: {overall_results[0]:.2f}%, "
              f"UA: {overall_results[1]:.2f}%")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
