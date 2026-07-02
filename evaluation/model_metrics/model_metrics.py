import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Column order for the shared tracking spreadsheet (mirrors the kick-off deck).
TRACKING_COLUMNS = [
    "model", "architecture", "learning_rate", "train_time_min",
    "accuracy", "precision", "recall", "f1", "notes",
]


def _predict_labels(model, generator):
    """
    True and predicted class indices for a shuffle=False generator.
    shuffle=False 
    """
    y_true = generator.classes
    y_pred = np.argmax(model.predict(generator), axis=1)
    return y_true, y_pred


def evaluate_model(model, generator, model_name):
    """
    Score a trained model and return one row for the tracking sheet:
    """
    y_true, y_pred = _predict_labels(model, generator)

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


def debug_model(model, generator, show_plot=True):
    """Print a per-class classification report, plot the confusion matrix as a
    heatmap, and return it as a labeled DataFrame (rows = true class,
    cols = predicted). Shared by every model so each one is debugged the exact
    same way. Pass show_plot=False to skip the figure (e.g. text-only runs)."""
    class_names = list(generator.class_indices.keys())
    y_true, y_pred = _predict_labels(model, generator)

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    if show_plot:
        _plot_confusion_matrix(cm, class_names)

    return pd.DataFrame(cm, index=class_names, columns=class_names)


def _plot_confusion_matrix(cm, class_names):
    """Render a confusion matrix as a Blues heatmap with the count written in
    each cell. Uses plain matplotlib (no seaborn) to avoid an extra dependency.
    Cell text flips to white on dark cells so it stays readable."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()


def plot_accuracy_comparison(csv_path="model_tracking.csv", metric="accuracy"):
    """Read the shared tracking sheet and draw a horizontal bar chart comparing
    every model on one metric (default: validation accuracy). Sorted best-last
    so the winner sits on top, with the value labelled on each bar."""
    df = pd.read_csv(csv_path).sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 1.5))
    bars = ax.barh(df["model"], df[metric], color="steelblue")
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1)
    ax.set_title(f"Model {metric} comparison (validation)")
    ax.bar_label(bars, fmt="%.3f", padding=3)

    fig.tight_layout()
    plt.show()
    return df


def record_result(row, csv_path="model_tracking.csv"):
    """Upsert one model's result into the shared tracking spreadsheet: read the
    existing CSV, drop any previous row for the same model name, append this
    one, and write it back. This lets each per-model notebook update the same
    sheet independently (no shared in-memory state needed). Returns the full
    tracking DataFrame so the calling cell can display it."""
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        prior = existing[existing["model"] == row["model"]]
        # A re-run that LOADS a saved model reports train_time_min=None. Keep the
        # time from the original training run instead of wiping it to blank.
        if row.get("train_time_min") is None and not prior.empty:
            row = {**row, "train_time_min": prior["train_time_min"].iloc[0]}
        existing = existing[existing["model"] != row["model"]]
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])

    combined = combined.reindex(columns=TRACKING_COLUMNS)
    combined.to_csv(csv_path, index=False)
    return combined
