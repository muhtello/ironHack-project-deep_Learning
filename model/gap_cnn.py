"""gap_cnn - CNN with a GAP head.

Name meaning: "gap" = GlobalAveragePooling2D, which replaces the baseline's
Flatten. It collapses each feature map to a single average value, shrinking the
dense head from ~184k to ~1.3k params - the main defense against overfitting.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_gap_cnn(input_shape, num_classes=10, dropout=0.5):
    # dropout is a parameter so variants can dial regularization up or down
    # (e.g. model 5 lowers it to 0.3 when combining GAP with augmentation).
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # GlobalAveragePooling2D replaces Flatten
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_gap_cnn(input_shape=(128, 128, 3), num_classes=10)
    model.summary()
