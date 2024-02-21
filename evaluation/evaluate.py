def evaluate(preds, targets):
    """
    Evaluate the model on the test set
    Args:
        preds: Predictions from the model as dictionary of {file_name: {"emotion"}}
        targets: True labels: dictionary of {file_name: {"emotion": emotion, "fold": fold}}

    """
    # TODO Average the metrics over the folds.

    # Find the intersection of the keys
    if set(preds.keys()) != set(targets.keys()):
        raise ValueError("Predictions and targets are not aligned")

    # Convert the dictionaries to lists
    keys = list(preds.keys())
    preds = [preds[key]["emotion"] for key in keys]
    targets = [targets[key]["emotion"] for key in keys]

    # Calculate the weighted accuracy
    weighted_accuracy = (preds == targets).sum() / len(preds)

    # Calculate the unweighted accuracy (average recall)
    unweighted_accuracy = (preds == targets).sum(axis=0) / len(preds)

    return weighted_accuracy, unweighted_accuracy
