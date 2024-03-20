import argparse
import json
import tabulate

import numpy as np


def corruption_error(baseline_errors, model_errors):
    """
    Calculate the Corruption Error (CE) Eq. 1 from the paper for a specific corruption type.
    Args:
        baseline_errors: np.array of baseline error rates
        model_errors: np.array of model error rates

    Returns:
        Corruption error
    """

    ce = sum(model_errors) / sum(baseline_errors) * 100
    return ce


def relative_corruption_error(baseline_errors, model_errors, model_clean_error, baseline_clean_error):
    """
    Calculate the Relative Corruption Error Eq. 3 from the paper for a specific corruption type.
    Args:
        baseline_errors: np.array of baseline error rates
        model_errors: np.array of model error rates
        model_clean_error: error rate of the model on clean data
        baseline_clean_error: error rate of the baseline on clean data
    Returns:
        Relative corruption error
    """

    rce = sum(model_errors - model_clean_error) / sum(baseline_errors - baseline_clean_error) * 100
    return rce


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model on the test set")
    parser.add_argument("-i", "--model_metrics", type=str, required=True,
                        help="Path to the model metrics json file with the error rates")
    parser.add_argument("-b", "--baseline_metrics", type=str, required=True,
                        help="Path to the baseline metrics json file with the error rates")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.model_metrics, "r") as f:
        model_metrics = json.load(f)

    with open(args.baseline_metrics, "r") as f:
        baseline_metrics = json.load(f)

    corruption_types = model_metrics.keys() - {"clean"}

    model_clean_error = model_metrics["clean"]
    baseline_clean_error = baseline_metrics["clean"]

    ces, rces = {}, {}
    for corruption_type in corruption_types:
        model_errors = np.array([model_metrics[corruption_type][key] for key in model_metrics[corruption_type]])
        baseline_errors = np.array(
            [baseline_metrics[corruption_type][key] for key in baseline_metrics[corruption_type]])
        ce = corruption_error(baseline_errors, model_errors)
        ces[corruption_type] = round(ce, 2)
        rce = relative_corruption_error(baseline_errors, model_errors, model_clean_error, baseline_clean_error)
        rces[corruption_type] = round(rce, 2)

    # ces and rces in the same table
    table = []
    for corruption_type in corruption_types:
        table.append([corruption_type, ces[corruption_type], rces[corruption_type]])

    print(tabulate.tabulate(table, headers=["Corruption Type", "CE %", "RCE %"]))

    print("---")
    print(f"Mean Corruption Error (mCE) %: {np.mean(list(ces.values())):.2f}")
    print(f"Relative mCE %: {np.mean(list(rces.values())):.2f}")


if __name__ == "__main__":
    main()
