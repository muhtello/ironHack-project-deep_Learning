# Task brief — section-by-section notes & status

Annotated version of the [project brief](https://github.com/ironhack-rmt-AI-student-materials/project-1-brief-CNN),
mapping each requirement to what we've built. Legend: ✅ done · 🟡 partial · ❌ todo

## Dataset
Chose **Animals-10** (~26k images, 10 classes). Italian folder names mapped 1:1
to English labels in `label_mapping.py`. ✅

## Assessment components

### 1. Data Preprocessing
- Normalization (rescale 1/255) ✅ · Resizing (128×128) ✅ · Augmentation
  (flip/rotate/shift/zoom, train-only) ✅ in model 4.
- **Visual representations of sample images + labels** ✅ — `plot_sample_images`
  (in `evaluation/plots/plots.py`) shows one random image per class with its
  English label; used in `model_1_baseline.ipynb` section 2b.

### 2. Model Architecture
- Conv + pooling + dense layers ✅ across baseline / GAP / GAP+BN.
- Multiple architectures compared (Flatten vs GAP vs BN) ✅.

### 3. Model Training
- Adam optimizer ✅ · Early stopping (`val_loss`, restore best weights) ✅.
- Overfitting prevention: Dropout, GlobalAveragePooling, BatchNorm, augmentation ✅.

### 4. Model Evaluation
- Validation set scoring ✅ · accuracy / precision / recall / F1 (macro) ✅.
- Confusion matrix visualization (heatmap) ✅.
- Test set reserved for the final winner only ✅ (`model_final_test.ipynb`).

### 5. Transfer Learning  ✅
- **Done** — `transfer_mobilenet` (`model/transfer_mobilenet.py`, `model_7`):
  frozen ImageNet-pretrained **MobileNetV2** base + a small GAP head (only ~13k
  trainable params). **Justification:** MobileNetV2 is lightweight (trains in
  ~4 min on CPU) yet its ImageNet features transfer well to animals.
- **Result:** val 0.940 / **test 0.943** — a +20-point jump over the best
  from-scratch model (0.739) and the overall winner. Also tried transfer +
  augmentation (`model_8`, 0.926) — augmentation didn't help here either.

### 6. Code Quality
- Organized into per-component folders, ≤150 lines/file, commented ✅.
- Shared `.py` logic + one notebook per model ✅.

### 7. Report (README)
- Architecture description ✅ · Preprocessing methodology ✅.
- Training specs (LR, batch size, epochs) ✅ — LR/batch size (32)/epochs all listed.
- Results + best-model justification ✅ — full 8-model table + winner rationale.
- Experimentation insights ✅ (BN early-stopping story; augmentation lost on both tracks).

### 8. Model Deployment  ✅
- **Done** — `app.py`: Gradio web demo serving the transfer winner. Upload a
  photo, get a live top-3 prediction. Runs locally with a temporary public
  `*.gradio.live` share link. (Brief lists this as optional; completed as a bonus.)

## Submission requirements
- Public GitHub repo ✅ (needs final push).
- `data/` with 5–10 test images ✅ (10, one per class).
- README documenting approach/findings ✅ draft.
- `requirements.txt` ✅.
- **Presentation slides** ❌ — not started.

## Plan

### Done
1. ✅ From-scratch model sweep (models 1–6) + winner `gap_cnn` (test 0.739).
2. ✅ README results table filled; `model_final_test.ipynb` run on the winner.
3. ✅ Batch size added to README training specs.
4. ✅ Transfer-learning model (component 5) — MobileNetV2 (model 7), test 0.943.
5. ✅ Transfer + augmentation experiment (model 8) — didn't beat plain transfer.
6. ✅ Sample-images-with-labels plot (component 1).
7. ✅ Deployment (component 8) — Gradio demo (`app.py`).

### Remaining
8. ❌ Presentation slides.
9. ❌ Final git push to the public repo.
