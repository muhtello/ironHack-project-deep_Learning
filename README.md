# Animals-10 Image Classification (CNN)

## About

An image classifier that recognizes **10 animal species** from a photo. Built
first as a **from-scratch Convolutional Neural Network** in TensorFlow/Keras,
then improved with **transfer learning** (MobileNetV2) — reaching **94% test
accuracy**. The project trains and compares 8 model variants, tracks every
experiment in a shared spreadsheet, and ships the winner as a live web demo.

**Sample prediction (Gradio demo):**

![Demo screenshot](sample_prediction\Demo.png)


**Live demo:** run `python app.py` to launch the Gradio app; it prints a local
URL plus a temporary public `*.gradio.live`

## Problem statement

- **Dataset:** [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
  — 26,179 images across 10 animal classes.
- **Task:** multi-class image classification (predict which of 10 animals is in
  a photo).
- **Goal:** build a CNN from scratch, systematically reduce overfitting, pick a
  winner honestly (validation for selection, test scored once), and see how far
  transfer learning can push accuracy beyond the from-scratch ceiling.

## Dataset

- **Source:** [Animals-10 on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- **Size:** 26,179 images · **Classes:** 10.
- **Classes** (Italian folder names mapped 1:1 to English):
  `dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel`.
- **Balance:** imbalanced — `spider`/`dog` dominate, `elephant`/`cat` are
  smaller. Handled with a stratified split and macro-averaged metrics.
- **License:** GPL 2 (per the Kaggle listing) — used here for educational purposes.
- **Note:** the dataset is **not committed** (exceeds GitHub's file-size limit).
  Place `archive.zip` at the project root and extract it to
  `extracted_animals/raw-img/` before running. A few sample images live in `data/`.

## Model architecture

Every model shares a conv/pool backbone; each variant changes **one thing** so
comparisons are fair. Model names are self-describing — each token is one
technique:

| Token | Stands for | Meaning |
|---|---|---|
| `baseline` | baseline | plain CNN, Flatten head, no extra tricks (the reference) |
| `gap` | GlobalAveragePooling2D | replaces Flatten; tiny dense head, less overfitting |
| `bn` | BatchNormalization | normalizes layer inputs after each conv |
| `aug` | data augmentation | random flips/rotations/shifts/zoom on training images |
| `transfer` | transfer learning | frozen ImageNet-pretrained MobileNetV2 base |

**From-scratch backbone:** 3 conv blocks (`Conv→Conv→MaxPool`, 32→64→128
filters) → head → `Dense(10, softmax)`. The key design choice was replacing
`Flatten` with `GlobalAveragePooling2D`, shrinking the dense head from ~184k to
~1.3k params (the main overfitting fix).

**Winner (transfer learning):** a frozen **MobileNetV2** (ImageNet weights) as a
feature extractor + `GlobalAveragePooling → Dropout → Dense(10)`. Only ~13k
trainable params; the `[0,1]→[-1,1]` rescale MobileNetV2 expects is baked into
the model so the shared data pipeline is unchanged.

### Preprocessing

- **Label mapping** — each Italian folder is mapped 1:1 to an English label,
  verified against the folders on disk at load time.
- **Split** — stratified 70% train / 15% validation / 15% test (`seed=42`).
- **Resize + normalize** — images resized to **128×128**, pixels rescaled to
  `[0, 1]` via `ImageDataGenerator(rescale=1/255)`.
- Validation/test generators use `shuffle=False` so predictions stay aligned
  with their labels.

### Training

- **Optimizer:** Adam (lr 0.001) · **Loss:** categorical crossentropy
- **Batch size:** 32 · **Image size:** 128×128 · **Epochs:** 20–60 ceiling
  (EarlyStopping on `val_loss` with `restore_best_weights=True` decides the real
  number).
- Models compared on **validation** metrics; the **test set is scored once**, on
  the final winner only (no leakage). Trained on CPU.

## Results

Validation metrics, macro-averaged (every class weighted equally — the dataset
is imbalanced):

| Model | Head | Val accuracy | Val F1 | Notes |
|---|---|---|---|---|
| baseline_cnn | Flatten | 0.699 | 0.668 | Overfits (~13% train-val gap) |
| gap_cnn | GlobalAveragePooling | 0.740 | 0.714 | **Best from-scratch model** — less overfit, higher accuracy |
| gap_bn_cnn | GAP + BatchNorm | 0.692 | 0.659 | BN hurt; needed patience=6 (BN makes val_loss noisy early) |
| gap_aug | GAP + augmentation | 0.668 | 0.630 | Augmentation over-regularized (with Dropout 0.5) |
| gap_aug2 | GAP + augmentation | 0.718 | 0.690 | Aug retry (Dropout 0.3, 60 epochs) — recovers most, still &lt; GAP |
| gap_deep | GAP + extra 256-block | 0.186 | 0.031 | Collapsed at lr=0.001 — plain deep CNN needs BN / lower LR |
| **transfer_mobilenet** | **MobileNetV2 (frozen) + GAP** | **0.940** | **0.932** | **Overall winner (bonus)** — transfer learning, ~13k trainable params, 4 min to train |
| transfer_mobilenet_aug | MobileNetV2 (frozen) + GAP + aug | 0.926 | 0.917 | Transfer + augmentation — slightly *below* plain transfer |

### Winner: test-set scores (scored once)

- **Best from-scratch — `gap_cnn`**: **test accuracy 0.739, macro-F1 0.709** —
  essentially identical to its 0.740 validation score (generalizes, not overfit).
  Weakest class: `cow` (recall 0.36, confused with `horse`/`sheep`).
- **Overall winner — `transfer_mobilenet`**: **test accuracy 0.943, macro-F1
  0.935** — a **+20-point** jump over the from-scratch ceiling, in **~4 minutes**
  (vs 32). Every weak class recovers: `cow` recall 0.36 → 0.82. This is the model
  served by the `app.py` demo.

**Insight — augmentation didn't help:** flip/rotate/shift/zoom was tested on
*both* tracks (`gap_aug`, `gap_aug2`, and `transfer_mobilenet_aug`). In every
case it landed *below* the un-augmented version, so the final winner uses none.

Per-class reports and **confusion-matrix heatmaps** are generated in each
notebook (`debug_model`) and for the winner in `model_final_test.ipynb`.

## Setup & installation

```bash
# 1. Clone
git clone <your-repo-url>
cd project-2

# 2. Create + activate a virtual environment
python -m venv .venv
.venv/Scripts/activate          # Windows (use source .venv/bin/activate on macOS/Linux)

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Notebooks) register the venv as a Jupyter kernel, then open any
#    model_*.ipynb and Run All
python -m ipykernel install --user --name project2-tf

# 5. (Demo) launch the web app
python app.py
```

Before running, place `archive.zip` at the project root and extract it to
`extracted_animals/raw-img/` (the dataset is not committed).

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
  transfer_mobilenet.py              model 7/8: transfer learning on a frozen MobileNetV2
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

## Tech stack

- **Language:** Python 3
- **Deep learning:** TensorFlow / Keras
- **ML utilities:** scikit-learn (metrics, split), pandas, NumPy
- **Visualization:** matplotlib
- **Demo / deployment:** Gradio
- **Notebooks:** Jupyter

## Future improvements

- **Fine-tune** the top MobileNetV2 layers (unfreeze) for a possible push past 94%.
- **Address class imbalance** directly (class weights / oversampling) to lift the
  remaining weaker classes.
- **Permanent hosting** (e.g. Hugging Face Spaces) so the demo runs without a
  local machine.
- Try other backbones (EfficientNet, ResNet50) and larger input sizes.

## Author

**Muhanad Tello** — Ironhack AI/ML project.
GitHub: (https://github.com/muhtello/ironHack-project-deep_Learning.git)
