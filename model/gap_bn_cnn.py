"""gap_bn_cnn - GAP head + BatchNorm.

Name meaning: "gap" = GlobalAveragePooling2D, "bn" = BatchNormalization 
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_gap_bn_cnn(input_shape, num_classes=10):
   
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_gap_bn_cnn(input_shape=(128, 128, 3), num_classes=10)
    model.summary()
