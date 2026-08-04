# Ordinal Methods for Monocular Depth Estimation

This directory contains the official implementation of the paper **"Ordinal Methods for Monocular Depth Estimation"** by Sofía Jimeno Lucía, supervised by Ricardo P. M. Cruz and Jaime S. Cardoso (FEUP / INESC TEC Porto).

---

## 📌 Overview

Monocular Depth Estimation (MDE) is key to 3D scene understanding. While discretizing continuous depth into ordinal bins improves robustness over direct regression, predicted depth maps can suffer from spatial inconsistency and abrupt artifacts.

This work introduces:
1. **Spatial Regularization Terms** (1-Side, 2-Sides, and Multi-Scale variants) added to ordinal loss functions to enforce local prediction smoothness across neighboring pixels.
2. **Probabilistic Neural Mapping (`OrdinalMapping`)**, a lightweight neural module that learns to convert predicted class probability distributions back into continuous depth values, outperforming standard expected value formulations.

---

## ⚙️ Architecture & Method

- **Backbone**: DeepLabv3 with a pre-trained ResNet-101 feature extractor.
- **Discretization**: Spacing-Increasing Discretization (SID) mapping depth into 128 logarithmic bins.
- **Spatial Regularization**: Penalizes neighboring pixel predictions falling into non-adjacent ordinal classes:
  - **1-Side**: Regularizes against immediate right and bottom neighbors.
  - **2-Sides**: Regularizes across both directions (left/right, top/bottom).
  - **Multi-Scale**: Applies regularization across downsampled resolutions ($0.5\times$, $0.25\times$) to enforce coarse and fine spatial coherence.

---

## 📁 Repository Structure

```text
├── IMCV_Sofia_Lucia_Thesis.pdf                         # Thesis document
├── Ordinal Methods for Monocular Depth Estimation.pdf  # Published conference/workshop paper
├── deep_ordinal.py                                     # Core ordinal loss functions & spatial regularization logic
├── models.py                                           # DeepLabv3 model and OrdinalMapping architecture wrappers
├── networks.py                                         # Neural network backbone components
├── train.py                                            # Main training script (dataset loading, loss setup, optimization)
└── train_test_mapping.py                               # Training and evaluation script for the continuous mapping module
