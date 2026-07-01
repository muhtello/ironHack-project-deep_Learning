import os
import sys

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_train_val_test_generators(
    df,
    project_root,
    image_size=(224, 224),
    batch_size=32,
    val_split=0.15,
    test_split=0.15,
    seed=42,
):
    """Split df into train/val/test, then wrap each in an ImageDataGenerator
    that resizes every image to image_size and rescales pixels to [0, 1].
    df['image_path'] must be relative to project_root."""
    train_df, remaining_df = train_test_split(
        df,
        test_size=val_split + test_split,
        stratify=df["label_en"],
        random_state=seed,
    )
    # test_split's share of remaining_df, so the 3-way split matches the
    # val_split/test_split ratios passed in, not an accidental 50/50 of the rest.
    remaining_test_share = test_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=remaining_test_share,
        stratify=remaining_df["label_en"],
        random_state=seed,
    )

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=project_root,
        x_col="image_path",
        y_col="label_en",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,
    )

    val_generator = eval_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=project_root,
        x_col="image_path",
        y_col="label_en",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,
        # No shuffle on val/test: keeps predictions aligned with each
        # generator's own labels/filenames, which evaluation needs later.
        shuffle=False,
    )

    test_generator = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=project_root,
        x_col="image_path",
        y_col="label_en",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "label_mapping"))
    from label_mapping import build_labeled_dataset, find_project_root

    root = find_project_root()
    dataset = build_labeled_dataset(project_root=root, save_csv=False)
    train_gen, val_gen, test_gen = build_train_val_test_generators(dataset, project_root=root)
    print(
        f"Train batches: {len(train_gen)}, "
        f"Val batches: {len(val_gen)}, "
        f"Test batches: {len(test_gen)}"
    )
    print(f"Classes: {train_gen.class_indices}")
