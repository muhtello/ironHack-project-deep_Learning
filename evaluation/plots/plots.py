import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import load_img


def plot_sample_images(df, project_root, seed=42):
    """Show one example image per class with its English label - the brief's
    'visual representation of sample images + labels' (data exploration). Picks
    a random image from each class so you see the raw data before any modeling.
    `df` is the labeled index (image_path, label_en); paths are root-relative."""
    class_names = sorted(df["label_en"].unique())

    cols = 5
    rows = int(np.ceil(len(class_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(seed)
    for ax, label in zip(axes, class_names):
        row = df[df["label_en"] == label].sample(1, random_state=int(rng.integers(1e6)))
        image_path = os.path.join(project_root, row["image_path"].iloc[0])
        ax.imshow(load_img(image_path))
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    # Blank any unused cells (if classes don't fill the grid).
    for ax in axes[len(class_names):]:
        ax.axis("off")

    fig.suptitle("Sample images per class", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_misclassified(model, generator, max_images=9):
    """Show a grid of images the model predicted WRONG, each titled
    'true -> pred'. This is the mandatory misclassified-images plot: it makes
    the confusion matrix concrete (e.g. cats that got called dogs). Works on any
    shuffle=False generator so predictions line up with generator.classes."""
    class_names = list(generator.class_indices.keys())

    generator.reset()
    y_true = generator.classes
    y_pred = np.argmax(model.predict(generator), axis=1)

    wrong = np.where(y_true != y_pred)[0][:max_images]
    if len(wrong) == 0:
        print("No misclassified images to show.")
        return

    cols = 3
    rows = int(np.ceil(len(wrong) / cols))
    target_size = generator.image_shape[:2]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, wrong):
        ax.imshow(load_img(generator.filepaths[idx], target_size=target_size))
        ax.set_title(
            f"true: {class_names[y_true[idx]]}\npred: {class_names[y_pred[idx]]}",
            fontsize=9,
        )
        ax.axis("off")

    # Blank any unused cells in the last row.
    for ax in axes[len(wrong):]:
        ax.axis("off")

    fig.suptitle("Misclassified examples", y=1.02)
    fig.tight_layout()
    plt.show()
