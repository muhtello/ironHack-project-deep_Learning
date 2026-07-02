"""transfer_mobilenet - transfer learning with a pre-trained MobileNetV2 base.

"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


def build_transfer_mobilenet(input_shape, num_classes=10, dropout=0.5):
    # Frozen ImageNet base = "feature extraction"
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,   # drop ImageNet's 1000-class head; we add our own
        weights="imagenet",
    )
    base.trainable = False

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        # The shared data_loader feeds pixels in [0, 1], but MobileNetV2's
        # weights expect [-1, 1]. Bake the rescale (x*2 - 1) into the model so
        # the data pipeline stays identical to every other model.
        layers.Rescaling(scale=2.0, offset=-1.0),
        base,
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
    model = build_transfer_mobilenet(input_shape=(128, 128, 3), num_classes=10)
    model.summary()
