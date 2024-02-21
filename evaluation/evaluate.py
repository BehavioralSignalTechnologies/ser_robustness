def evaluate(preds, targets):
    """
    Evaluate the model on the test set
    Args:
        preds: Predictions from the model as dictionary of {file_name: {"emotion"}}
        targets: True labels: dictionary of {file_name: {"emotion": emotion, "fold": fold}}

    """

    if set(preds.keys()) != set(targets.keys()):
        raise ValueError("Predictions and targets have different keys")

    if len(preds) != len(targets):
        raise ValueError(f"Predictions and targets have different lengths: {len(preds)} and {len(targets)}"

        sessions = set([target["fold"] for target in targets.values()])
        results = {}
        for session in sessions:
            keys = [key for key in targets.keys() if targets[key]["fold"] == session]

        # Convert the dictionaries to lists
        keys = list(preds.keys())
        preds = [preds[key]["emotion"] for key in keys]
        targets = [targets[key]["emotion"] for key in keys]

        # Calculate the weighted accuracy
        weighted_accuracy = (preds == targets).sum() / len(preds)

        # Calculate the unweighted accuracy (average recall)
        unweighted_accuracy = (preds == targets).sum(axis=0) / len(preds)

        results[session] = [weighted_accuracy, unweighted_accuracy]

        results["average"] = [sum([result[0] for result in results.values()]) / len(results),
                              sum([result[1] for result in results.values()]) / len(results)]
    return results
