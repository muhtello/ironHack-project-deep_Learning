"""Gradio demo: upload an animal photo, the trained CNN predicts its class.

Wraps the winning model (transfer_mobilenet) in a tiny web UI so anyone can try it in a
browser - no notebook, no code. Run `python app.py` and open the local URL it
prints (add share=True for a temporary public link).
"""

import numpy as np
from PIL import Image
from tensorflow import keras

import gradio as gr

# The best model overall: transfer learning on a frozen MobileNetV2 base
# (~0.94 validation accuracy) - far better than the from-scratch CNNs. Its
# baked-in [0,1]->[-1,1] rescale means the preprocessing below is unchanged.
MODEL_PATH = "models_saved/transfer_mobilenet.keras"

# The model's input size, fixed at train time (128x128 RGB).
IMAGE_SIZE = (128, 128)

# CRITICAL: the softmax outputs are ordered exactly how Keras ordered the
# classes during training. The generators used y_col="label_en", so Keras
# sorted the ENGLISH label strings alphabetically - NOT the Italian folder
# order. Getting this list wrong silently mislabels every prediction.
CLASS_NAMES = sorted(
    [
        "dog", "horse", "elephant", "butterfly", "chicken",
        "cat", "cow", "sheep", "spider", "squirrel",
    ]
)

model = keras.models.load_model(MODEL_PATH)


def predict(image):
    """Preprocess one uploaded PIL image exactly like the eval pipeline
    (resize to 128x128, rescale to [0, 1]) and return a {label: confidence}
    dict. Gradio's Label component turns that into a ranked bar chart."""
    if image is None:
        return {}

    # Match training preprocessing: RGB, 128x128, pixels in [0, 1].
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype="float32") / 255.0
    batch = np.expand_dims(array, axis=0)  # model expects a batch dimension

    probabilities = model.predict(batch, verbose=0)[0]
    return {name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an animal photo"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Animals-10 Classifier (MobileNetV2)",
    description=(
        "Transfer learning on a MobileNetV2 base (~94% test accuracy) for the "
        "Animals-10 dataset. Upload a photo of one of these animals: "
        + ", ".join(CLASS_NAMES) + "."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    # share=True opens a temporary public *.gradio.live URL (valid ~72h) that
    # tunnels to this machine - lets anyone try the demo without a deploy.
    demo.launch(share=True)
