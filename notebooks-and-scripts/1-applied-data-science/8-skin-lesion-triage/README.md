# Medical Image Analysis for Skin Lesion Triage

This repository contains the implementation files accompanying the portfolio project **Medical Image Analysis for Skin Lesion Triage**, part of the book *Data Science Across Domains: From Methods and Models to Business Applications*.

The project investigates multiclass skin lesion image classification using deep learning models for computer-assisted triage. Rather than focusing solely on predictive performance, the implementation follows a complete and reproducible machine learning workflow encompassing data preparation, leakage-aware dataset partitioning, model development, model comparison, evaluation, calibration, and error analysis.

Two complementary modelling strategies are implemented and compared:

* **Compact Convolutional Neural Network (CNN)** developed from scratch as a lightweight baseline architecture.
* **Transfer Learning with ResNet-18**, adapted from a model pretrained on ImageNet and fine-tuned for multiclass skin lesion classification.

Model selection is performed using validation macro F1 score, followed by independent evaluation on a held-out test set. The workflow also includes probability calibration through temperature scaling, selective prediction based on confidence thresholds, confusion analysis, and inspection of high-confidence prediction errors.

## Repository Structure

* `notebook.ipynb` — Complete end-to-end implementation of the modelling workflow.
* `data/` — Input datasets required to reproduce the analyses.

During execution, the notebook automatically generates figures, tables, metadata files, trained model checkpoints, and other reproducibility artefacts.

## Dataset

The analyses are based on the **ISIC 2019 Challenge Dataset**, released by the International Skin Imaging Collaboration (ISIC).

The official dataset can be obtained from the ISIC Archive collection [here](https://api.isic-archive.com/collections/66/).

The implementation expects the image files together with the corresponding metadata and ground-truth labels to be placed inside the local `data/` directory before executing the notebook.

The dataset is distributed by the ISIC Archive under its own licensing terms and is **not** included in this repository.

## Implementation

The notebook implements a fully reproducible deep learning pipeline including:

* lesion-level train/validation/test partitioning to prevent data leakage;
* exploratory analysis of class distributions;
* image preprocessing and data augmentation;
* training of both a custom CNN and a transfer-learning ResNet-18 model;
* weighted loss functions to mitigate class imbalance;
* learning-rate scheduling, gradient clipping, and early stopping;
* validation-based model selection;
* evaluation using accuracy, balanced accuracy, macro F1, and weighted F1;
* class-level performance analysis and confusion matrices;
* confidence analysis and inspection of high-confidence errors;
* probability calibration using temperature scaling;
* reliability diagrams and expected calibration error;
* selective prediction analysis based on confidence thresholds;
* automatic export of figures, tables, metadata, and trained model checkpoints.

## Companion Chapter

The theoretical background, methodology, implementation details, experimental results, and discussion are presented in the corresponding chapter of *Data Science Across Domains: From Methods and Models to Business Applications*.

The notebook contained in this repository provides a fully reproducible computational workflow supporting the analyses presented in the chapter.

## Copyright

Copyright © Rodrigo Kang.

All rights reserved.