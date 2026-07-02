"""gap_deep_cnn - deeper GAP model.

Name meaning: "gap" = GlobalAveragePooling2D head, "deep" = one extra conv block
added on top of gap_cnn. More capacity to see if
the plain GAP model was leaving accuracy on the table (it still had an ~8%
train-val gap).
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_gap_deep_cnn(input_shape, num_classes=10, dropout=0.5):
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
        # Extra 4th block (256 filters) - the single change vs gap_cnn.
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
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
    model = build_gap_deep_cnn(input_shape=(128, 128, 3), num_classes=10)
    model.summary()
