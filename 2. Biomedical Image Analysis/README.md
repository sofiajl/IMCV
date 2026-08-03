# 2. Biomedical Image Analysis (BMIA)

This directory contains code, documentation, and experimental results for the Biomedical Image Analysis module.


## 📚 Module Overview

The BMIA module focuses on the practical application of computer vision, pattern recognition, and deep learning techniques to medical and biological imaging modalities (e.g., OCT, MRI, CT).

Medical image pre-processing, classical feature engineering, semantic segmentation, class imbalance handling, model evaluation, and cross-validation strategies.

More details: [https://imcv.eu/guide/2023-2024/bmia/](https://imcv.eu/guide/2023-2024/bmia/)



## 📁 Cases Overview

### 🔹 Case 1: DME Image Classification Methodology
* **Objective:** Automatic classification of Optical Coherence Tomography (OCT) images to detect the presence or absence of Diabetic Macular Edema (DME).
* **Dataset:** 100 rectangular OCT images (50 DME vs. 50 Normal).
* **Feature Extraction (126 features extracted):**
  * **Local Binary Patterns (LBP):** 20 features.
  * **Gray Level Co-occurrence Matrix (GLCM):** 6 statistical features (contrast, homogeneity, energy, etc.).
  * **Gabor Filters:** 50 features capturing mean amplitude and local energy across multi-orientations/frequencies.
  * **Principal Component Analysis (PCA):** 20 direct image features.
  * **Histogram of Oriented Gradients (HOG) + PCA:** 20 features extracted via PCA over HOG descriptors.
* **Feature Selection:** Recursive Feature Elimination (RFE) with a Random Forest estimator reducing the feature space from 126 to 30 key features.
* **Selected Classifier:** Random Forest Classifier.


### 🔹 Case 2: Pathological Fluid Segmentation in DME
* **Objective:** Automated segmentation of pathological fluid regions in Optical Coherence Tomography (OCT) scans for Diabetic Macular Edema (DME).
* **Dataset:** 50 OCT scans and 50 corresponding ground truth binary masks.
* **Architectures Evaluated:**
  1. **Attention U-Net**
  2. **U-Net + ResNet-34** (*Discarded due to extremely poor performance/near-zero scores*)
  3. **LinkNet + Inception-V3**
* **Evaluation Metrics:** Dice Similarity Coefficient, Intersection over Union (IoU), and Thresholded IoU / F-score.
