import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_model(model, generator, model_name):
    """Score a trained model on a (shuffle=False) generator and return one row
    for the model-comparison sheet: accuracy plus macro-averaged precision,
    recall and f1. Macro (not weighted) so every class counts equally, which
    matters here because the dataset is imbalanced (dog/spider dominate)."""
    # shuffle=False on the generator means predictions come back in the same
    # order as generator.classes, so true and predicted labels line up.
    y_true = generator.classes
    y_pred = np.argmax(model.predict(generator), axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
