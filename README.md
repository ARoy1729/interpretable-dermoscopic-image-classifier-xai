# Explainable Skin Lesion Classification

The project uses a single **DenseNet121** model with ImageNet transfer learning to classify the seven HAM10000 diagnostic categories. The classification pipeline uses **class-weighted cross-entropy** to counter the strongly imbalanced label distribution. At inference time, every image gets two complementary explanations:

- **Grad-CAM** — highlights spatial regions associated with the predicted class at the final convolutional feature stage.
- **SHAP (GradientExplainer)** — attributes the predicted class score back toward the input pixels using expected gradients.

The Streamlit app puts those pieces together in one small interface: upload an image → get the predicted class and confidence → inspect Grad-CAM and SHAP overlays → read a short model-behaviour explanation.

> **Scope:** this is an educational/research portfolio project, not a medical device and not a diagnostic tool.

---

## Why I built it

A plain image classifier can look impressive while still being hard to trust. A portfolio project is more interesting when it demonstrates the complete reasoning loop:

**data preparation → imbalance handling → transfer learning → evaluation → visual explanation → failure analysis → interactive inference**

The goal here is not to benchmark ten architectures. It is to take **one sensible CNN architecture and execute the pipeline carefully enough that the mistakes are inspectable too.**

---

## What the pipeline contains

```text
HAM10000 metadata + images
          │
          ▼
  patient-aware train/val/test split
          │
          ▼
  resize + augmentation + normalization
          │
          ▼
 DenseNet121 (ImageNet weights)
          │
          ├── weighted cross-entropy
          ├── macro-F1 monitoring
          └── early stopping
          │
          ▼
       best checkpoint
          │
          ├── classification metrics
          ├── confusion matrix
          ├── one-vs-rest ROC/AUC
          ├── misclassification review
          └── Streamlit inference
                         │
                         ├── prediction + confidence
                         ├── Grad-CAM
                         └── SHAP
```

### Seven classes

| Code | Diagnosis |
|---|---|
| `akiec` | Actinic keratoses / Bowen's disease |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

---

## Repository structure

```text
explainable-skin-lesion-classifier/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── README.md
│   └── ... HAM10000 files live here ...
├── models/
│   ├── README.md
│   └── densenet121_ham10000.pt        # generated after training
├── outputs/
│   ├── README.md
│   ├── metrics.json                    # generated
│   ├── confusion_matrix.png            # generated
│   ├── roc_curves.png                  # generated
│   ├── test_predictions.csv            # generated
│   └── failure_examples/               # generated
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── explainability.py
    ├── metrics.py
    ├── model.py
    └── utils.py
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd explainable-skin-lesion-classifier
```

### 2. Create an environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch, use the wheel that matches the machine's CUDA/CPU setup when needed. The project code only relies on the stable `torch` / `torchvision` APIs exposed by the requirements above.

### 4. Add HAM10000

Put the metadata CSV and the two image directories under `data/`. The expected layout is documented in [`data/README.md`](data/README.md).

### 5. Build lesion-aware splits

```bash
python scripts/preprocess.py
```

The split step keeps images from the same lesion together. This is an intentional choice: leakage through related patient images can make validation performance look cleaner than it really is.

### 6. Train

```bash
python scripts/train.py
```

Useful overrides:

```bash
python scripts/train.py --epochs 20 --batch-size 32 --lr 3e-4
```

Training saves the best checkpoint by validation macro-F1 to:

```text
models/densenet121_ham10000.pt
```

### 7. Evaluate and generate failure examples

```bash
python scripts/evaluate.py
```

This produces:

- accuracy, macro precision, macro recall and macro F1
- multiclass one-vs-rest ROC-AUC
- confusion matrix image
- ROC curve image
- test-set prediction CSV
- three high-confidence misclassification examples with Grad-CAM overlays

---

## Evaluation results

The repository is designed so the actual values are generated from my own run rather than written into the README ahead of time.

After `python scripts/evaluate.py`, copy the values from `outputs/metrics.json` into the table below:

| Metric | Test score |
|---|---:|
| Accuracy | `XX.XX%` |
| Macro Precision | `XX.XX%` |
| Macro Recall | `XX.XX%` |
| Macro F1 | `XX.XX%` |
| Macro ROC-AUC (OvR) | `XX.XX%` |

Per-class precision / recall / F1 are also saved under `metrics.json` so the weaker rare classes are not hidden by one overall accuracy number.

---

## Failure analysis

The evaluation script automatically selects **2–3 high-confidence mistakes** and produces a small visual review set in:

```text
outputs/failure_examples/
```

For each example, the generated image shows the original lesion beside the Grad-CAM overlay together with the true label, predicted label and confidence.

When I review those cases, the main questions are:

1. Did the model focus on the lesion, or on irrelevant artifacts such as hair / ruler marks / image borders?
2. Is the error clinically understandable because the competing categories share visual traits?
3. Does the model become overconfident on a visually ambiguous case?

That last point matters more than chasing a single headline metric: a useful classifier should also have mistakes that I can inspect and explain.

---

## Explainability design

### Grad-CAM

Grad-CAM is computed from the final DenseNet121 dense block. The implementation lives in `src/explainability.py` so the same logic is reused by the evaluation script and the Streamlit app.

The map is class-specific: it backpropagates the logit of the predicted class, averages gradients across spatial positions, and uses those weights to form a coarse localization map. The map is then resized and blended with the original image.

### SHAP

The app uses `shap.GradientExplainer` with a small deterministic neutral background. Only the **predicted class** is explained at inference time because explaining all seven outputs interactively would make the demo unnecessarily slow.

The result is converted into a spatial attribution map by aggregating absolute attributions across RGB channels. This gives a directly comparable visual overlay without pretending the two methods mean the exact same thing.

That distinction is intentional: **Grad-CAM is feature-map based; SHAP is input-attribution based.** Comparing them is part of the point of the project.

---

## Streamlit app

Run:

```bash
streamlit run app.py
```

The app provides:

- predicted HAM10000 class
- confidence score
- uploaded image
- Grad-CAM overlay
- SHAP overlay
- class probability chart
- natural-language interpretation of the model behaviour

### Screenshots

Add the actual screenshots from your run here. Keeping them in `assets/screenshots/` makes the README self-contained for GitHub.

```text
assets/screenshots/
├── app-overview.png
├── gradcam-vs-shap.png
└── failure-analysis.png
```

Suggested README placements once captured:

`app-overview.png`

`gradcam-vs-shap.png`

`failure-analysis.png`

---

## Design decisions I would call out in an interview

**Why DenseNet121?** Dense connections give the network strong feature reuse while keeping the project within a reasonable compute budget for a student/developer workflow. I only use one backbone here because the goal is a coherent explainable pipeline, not an architecture bake-off.

**Why weighted loss instead of only oversampling?** The seven HAM10000 labels are not evenly distributed. A weighted cross-entropy loss directly changes the training objective so errors on under-represented classes matter more without duplicating minority images in every epoch.

**Why macro-F1 for model selection?** Accuracy can be dominated by the most common diagnosis. Macro-F1 gives each class equal weight when deciding which checkpoint is worth keeping.

**Why lesion-aware splitting?** The dataset contains multiple images associated with some patients. Keeping those groups together reduces the chance that visually related examples leak across train and evaluation sets.

**Why two explanation methods?** A single heatmap can be persuasive without being informative. Showing Grad-CAM next to SHAP makes it easier to see whether two different attribution mechanisms point toward similar image evidence.

---

## Limitations

This is still a portfolio-scale research implementation. It does not solve clinical calibration, domain shift across cameras or institutions, external validation, lesion segmentation, demographic subgroup analysis, uncertainty estimation, or prospective clinical evaluation.

The explainability maps should also be treated as **evidence of model behaviour, not proof of causal reasoning**.

---

## Suggested Git history

The repository can be built in small, believable iterations rather than one giant “initial commit”:

```text
feat: add HAM10000 patient-aware preprocessing
feat: add weighted DenseNet121 training pipeline
feat: add macro-F1 checkpoint selection
feat: add test metrics and confusion matrix reporting
feat: add Grad-CAM explanation utility
feat: add SHAP expected-gradient explanations
feat: add automated misclassification review
feat: build Streamlit inference app
docs: add project architecture and setup guide
chore: clean generated artifacts from git
```

---

## Reproducibility notes

- Random seed is fixed at `42` in `src/config.py`.
- The training checkpoint stores the class order used by inference.
- Image preprocessing is centralized in `src/data.py` / `src/explainability.py` to avoid subtle train-vs-app preprocessing drift.
- Generated outputs are kept separate from source code.

---

## License and dataset note

Before publishing a trained checkpoint or screenshots containing dataset examples, verify that your use of the HAM10000 dataset follows its stated terms and the terms of the source from which you obtained it.

The source code in this repository can be licensed separately from the dataset and from any trained weights you choose to publish.
