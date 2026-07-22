# Skin Lesion Triage

This repository contains the implementation accompanying the portfolio project **Skin Lesion Triage**, part of the book *Data Science Across Domains: From Methods and Models to Business Applications*.

The project investigates deep learning methods for computer-assisted skin lesion triage using dermoscopic images. Rather than attempting automated diagnosis, the objective is to identify lesions requiring **priority clinical review** within a transparent and reproducible machine learning workflow.

The implementation emphasises sound experimental design alongside predictive performance. The workflow covers data preparation, leakage-aware dataset partitioning, model development, validation-based model selection, independent testing, robustness assessment, subgroup analysis, and post-hoc interpretability.

Two complementary modelling strategies are implemented and compared:

- **BaselineConvNet** — a compact convolutional neural network developed from scratch as a lightweight baseline.
- **ResNet-18 Transfer Learning** — a pretrained ResNet-18 progressively fine-tuned for the binary triage task.

Model selection is performed **exclusively on the validation partition** using **Average Precision (AP)**. After selecting the best-performing model, the operating threshold is determined on the validation set by maximising the **F₂ score**, followed by a single frozen evaluation on the held-out test set.

---

# Repository Structure

```text
.
├── README.md
├── requirements.txt
│
├── data/
│   ├── ISIC-images/
│   └── metadata.csv
│
└── python/
    ├── 1-skin-lesion-triage.ipynb
    └── output/
        ├── figures/
        ├── models/
        ├── predictions/
        ├── reports/
        └── tables/
```

- **README.md** — project overview and execution instructions.
- **requirements.txt** — Python package dependencies.
- **data/** — dermoscopic images and metadata required to reproduce the analyses.
- **python/** — notebook implementation.
- **python/output/** — automatically generated artefacts created during notebook execution.

---

# Dataset

The implementation uses the **HAM10000** dermoscopic image dataset distributed through the **ISIC 2018 Task 3** training collection.

The dataset is **not** included in this repository.

Before running the notebook, place:

- all dermoscopic images inside:

```text
data/ISIC-images/
```

- the metadata file as:

```text
data/metadata.csv
```

The notebook located in `python/` reads the dataset directly from the repository-level `data/` directory.

The binary triage target used throughout this project is derived exclusively for analytical purposes and should not be interpreted as an original dataset label or as a clinically validated referral protocol.

---

# Implementation

The notebook implements a fully reproducible deep learning workflow including:

- lesion-aware train, validation, and test partitioning to prevent information leakage;
- exploratory analysis of lesion and class distributions;
- train-only image normalisation;
- data augmentation applied exclusively to the training partition;
- implementation of a compact CNN baseline;
- implementation of a transfer-learning ResNet-18 model;
- weighted binary cross-entropy to mitigate class imbalance;
- learning-rate scheduling, gradient clipping, and early stopping;
- validation-based checkpoint selection;
- validation-based operating-threshold selection;
- independent frozen test evaluation;
- lesion-aware bootstrap confidence intervals;
- subgroup performance analysis;
- Grad-CAM visualisations for qualitative model interpretation;
- automatic export of figures, tables, trained models, prediction files, and reproducibility artefacts.

---

# Reproducibility

Running `python/1-skin-lesion-triage.ipynb` automatically generates the artefacts required to reproduce the complete experimental workflow inside `python/output/`.

The exported artefacts include:

### Baseline experiment

- training summary;
- final test metrics;
- reproducibility manifest;
- project summary.

### Final selected model

- trained ResNet-18 checkpoint;
- prediction files;
- evaluation summary;
- final test metrics;
- reproducibility manifest;
- project summary;
- bootstrap confidence intervals;
- subgroup analyses;
- comparison tables;
- figures generated throughout the evaluation pipeline.

By preserving both the baseline experiment and the final selected model, the repository maintains a complete and traceable record of model development, selection, and evaluation.

---

# Companion Chapter

The theoretical background, modelling decisions, implementation details, experimental results, and discussion are presented in the corresponding chapter of *Data Science Across Domains: From Methods and Models to Business Applications*.

This repository provides the complete computational workflow supporting the analyses presented in the chapter.

---

# Requirements

Install the required Python packages before running the notebook:

```bash
pip install -r requirements.txt
```

---

# Copyright

Copyright © Rodrigo Kang.

All rights reserved.