# Animals-10 Image Classification (CNN)

Deep-learning image classifier for the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
dataset (26,179 images, 10 classes) built with a from-scratch Convolutional
Neural Network in TensorFlow/Keras. The project trains and compares several CNN
variants, tracking every experiment in a shared spreadsheet, and reports the
winner on a held-out test set.

## Classes

The dataset's folders are named in Italian; they are mapped to English labels:

`dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel`

## Project structure

```
preprocessing/
  label_mapping/label_mapping.py     Italian->English mapping, builds the image index
  data_loader/data_loader.py         stratified 70/15/15 split + image generators
model/
  baseline_cnn.py                    model 1: Conv blocks -> Flatten head
  gap_cnn.py                         model 2/4/5: GlobalAveragePooling head (dropout param)
  gap_bn_cnn.py                      model 3: GAP + BatchNorm
  gap_deep_cnn.py                    model 6: GAP + extra 256-filter conv block
  transfer_mobilenet.py              model 7: transfer learning on a frozen MobileNetV2
evaluation/
  model_metrics/model_metrics.py     evaluate / debug / record + confusion matrix
  plots/plots.py                     sample-images grid + misclassified-images grid
model_1_baseline.ipynb               one notebook per model, shared .py logic
model_2_gap.ipynb
model_3_gap_bn.ipynb
model_4_augmentation.ipynb           gap_cnn + augmentation (dropout 0.5)
model_5_gap_aug2.ipynb               gap_cnn + augmentation retry (dropout 0.3, 60 epochs)
model_6_gap_deep.ipynb               deeper GAP (collapsed at lr=0.001 - see results)
model_7_transfer.ipynb               transfer learning (MobileNetV2) - the overall winner
model_8_transfer_aug.ipynb           transfer learning + augmentation (didn't beat model 7)
app.py                               Gradio demo: upload a photo, get a live prediction
model_final_test.ipynb               scores the winning model once on the test set
model_tracking.csv                   the experiment log (one row per run)
model_comparison.csv                 ranked summary of all models for the report
models_saved/                        trained .keras models (winner reused by final test)
data/                                sample images (one per class)
```

## Model naming convention

Model names are self-describing: each token is one technique, stacked
left-to-right, so the name tells you what's inside without reading the code.

| Token | Stands for | Meaning |
|---|---|---|
| `baseline` | baseline | plain CNN, Flatten head, no extra tricks (the reference) |
| `gap` | GlobalAveragePooling2D | replaces Flatten; tiny dense head, less overfitting |
| `bn` | BatchNormalization | normalizes layer inputs after each conv |
| `aug` | data augmentation | random flips/rotations/shifts/zoom on training images |
| `cnn` | Convolutional Neural Network | the model type |

Examples: `gap_cnn` = GAP head · `gap_bn_cnn` = GAP + BatchNorm · `gap_aug` =
GAP + augmentation. Each model changes **one thing** from the previous, and the
name grows by one token to reflect it.

## Preprocessing

- **Label mapping** — each Italian folder name is mapped 1:1 to an English label;
  the mapping is verified against the folders on disk at load time.
- **Split** — stratified 70% train / 15% validation / 15% test (`seed=42`), so
  class balance is preserved across all three sets.
- **Resize + normalize** — every image is resized to **128x128** and pixels are
  rescaled to `[0, 1]` via `ImageDataGenerator(rescale=1/255)`.
- Validation and test generators use `shuffle=False` so predictions stay aligned
  with their labels for evaluation.

## Models & training

- **Optimizer:** Adam (lr 0.001) · **Loss:** categorical crossentropy
- **Batch size:** 32 · **Image size:** 128×128 · **Epochs:** 20–60 ceiling
  (EarlyStopping decides the real number).
- **EarlyStopping** on `val_loss` (`restore_best_weights=True`); `epochs` is a
  ceiling, not a target.
- Models are **compared on validation metrics**; the **test set is scored once**,
  on the final winner only (no leakage).
- Trained on CPU (TensorFlow has no native-Windows GPU support).

## Results (validation, macro-averaged)

| Model | Head | Val accuracy | Val F1 | Notes |
|---|---|---|---|---|
| baseline_cnn | Flatten | 0.699 | 0.668 | Overfits (~13% train-val gap) |
| gap_cnn | GlobalAveragePooling | 0.740 | 0.714 | **Best from-scratch model** — less overfit, higher accuracy |
| gap_bn_cnn | GAP + BatchNorm | 0.692 | 0.659 | BN hurt; needed patience=6 (BN makes val_loss noisy early) |
| gap_aug | GAP + augmentation | 0.668 | 0.630 | Augmentation over-regularized (with Dropout 0.5) |
| gap_aug2 | GAP + augmentation | 0.718 | 0.690 | Aug retry (Dropout 0.3, 60 epochs) — recovers most, still &lt; GAP |
| gap_deep | GAP + extra 256-block | 0.186 | 0.031 | Collapsed at lr=0.001 — plain deep CNN needs BN / lower LR |
| **transfer_mobilenet** | **MobileNetV2 (frozen) + GAP** | **0.940** | **0.932** | **Overall winner (bonus)** — transfer learning, only ~13k trainable params, 4 min to train |
| transfer_mobilenet_aug | MobileNetV2 (frozen) + GAP + aug | 0.926 | 0.917 | Transfer + augmentation — slightly *below* plain transfer (aug didn't help here either) |

### From-scratch winner vs. transfer-learning bonus

Two winners, two stories:

- **Best from-scratch model — `gap_cnn`**: scored once on the held-out test set
  (see `model_final_test.ipynb` history) at **accuracy 0.739, macro-F1 0.709**,
  essentially identical to its 0.740 validation score. Its weakest class was
  `cow` (recall 0.36, confused with `horse`/`sheep`).
- **Overall winner — `transfer_mobilenet` (bonus)**: reusing a frozen
  ImageNet-pretrained MobileNetV2 base and training only a small head lifts the
  held-out **test accuracy to 0.943 (macro-F1 0.935)** — a **+20-point** jump
  over the from-scratch ceiling, trained in **~4 minutes** (vs 32). Every weak
  class recovers: `cow` recall goes 0.36 → 0.84. This is the model served by the
  `app.py` demo.

**Note on augmentation:** data augmentation (flip/rotate/shift/zoom) was tested
on *both* tracks — the from-scratch GAP model (`gap_aug`, `gap_aug2`) and the
transfer model (`transfer_mobilenet_aug`, 0.926). In every case it landed
*below* the un-augmented version, so it was not used in the final winner. On
this dataset the plain pipeline generalizes better than the augmented one.

## How to run

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt

# Register the venv as a Jupyter kernel, then open any model_*.ipynb and Run All.
python -m ipykernel install --user --name project2-tf
```

The dataset itself is not committed (it exceeds GitHub's file-size limit). Place
`archive.zip` at the project root and extract it to `extracted_animals/raw-img/`
before running.
