import os
import sys
import argparse

from sklearn.metrics import accuracy_score, recall_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from parsing.iemocap import ParserForIEMOCAP


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
    Evaluate the model on the test set
    Args:
        preds: Predictions from the model as dictionary of {file_name: {"emotion": emotion}}
        targets: True labels: dictionary of {file_name: {"emotion": emotion, "fold": fold}}
    """

    # Remove "other" emotion from the predictions and targets
    preds = {key: value for key, value in preds.items() if value["emotion"] != "other"}
    targets = {key: value for key, value in targets.items() if value["emotion"] != "other"}

    # Check that the classes are valid
    classes = ["neutral", "happy", "sad", "angry"]
    pred_classes = set([pred["emotion"] for pred in preds.values()])
    target_classes = set([target["emotion"] for target in targets.values()])
    if pred_classes != classes:
        raise ValueError(f"Predictions contain invalid classes: {pred_classes}")
    if target_classes != classes:
        raise ValueError(f"Targets contain invalid classes: {target_classes}")

    # Check that the filenames are the same
    if set(preds.keys()) != set(targets.keys()):
        raise ValueError("Predictions and targets have different keys")

    # Check that the lengths are the same
    if len(preds) != len(targets):
        raise ValueError(f"Predictions and targets have different lengths: {len(preds)} and {len(targets)}")

    # Evaluate with 5-fold cross-validation
    sessions = set([target["fold"] for target in targets.values()])
    results = {}
    for session in sessions:
        fold_filenames = [key for key, value in targets.items() if value["fold"] == session]

        fold_preds = [preds[key]["emotion"] for key in fold_filenames]
        fold_targets = [targets[key]["emotion"] for key in fold_filenames]

        # Calculate the weighted accuracy
        weighted_accuracy = accuracy_score(fold_targets, fold_preds)

        # Calculate the unweighted accuracy (macro averaged recall)
        unweighted_accuracy = recall_score(fold_targets, fold_preds, average="macro")

        results[session] = [weighted_accuracy, unweighted_accuracy]

        results["average"] = [sum([result[0] for result in results.values()]) / len(results),
                              sum([result[1] for result in results.values()]) / len(results)]
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model on the test set")
    parser.add_argument("-csv", "--predictions", type=str, required=True, help="Path to the predictions CSV file")
    parser.add_argument("-p", "--data_path", type=str, required=True, help="Path to the dataset path")
    parser.add_argument("-d", "--dataset", type=str, choices=["iemocap"], required=True, help="Dataset to evaluate")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    preds = parse_csv(args.predictions)
    if args.dataset == "iemocap":
        parser = ParserForIEMOCAP(args.data_path)
        targets = parser.run_parser()
        targets = {os.path.basename(k): v for k, v in targets.items()}
        results = evaluate_iemocap(preds, targets)
        print(f"Weighted accuracy: {results['average'][0]:.2f}")
        print(f"Unweighted accuracy: {results['average'][1]:.2f}")
        for session, (weighted_accuracy, unweighted_accuracy) in results.items():
            print(f"Session {session}: Weighted accuracy: {weighted_accuracy:.2f}, "
                  f"Unweighted accuracy: {unweighted_accuracy:.2f}")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
