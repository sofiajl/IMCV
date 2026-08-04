
# IMCV — International Master in Computer Vision & Image Processing

Welcome to the **IMCV** repository! This repository serves as a personal archive and showcase for the practical coursework, laboratory cases, practicum projects, and thesis research conducted during my time in the International Master in Computer Vision and Image Processing.

## 📌 Repository Overview

This repository houses a collection of computer vision projects (code implementations, Jupyter Notebooks, experimental pipelines, and theoretical analysis) across different modules of the IMCV program: biomedical vision, human activity recognition, digital filtering, motion detection, optical flow estimation, single-object tracking, multi-object tracking, facial recognition login system, as part of a collaboration within a company, and my master's thesis.


## 📂 Modules & Directory Structure

```text
IMCV/
├── 1. Advance Image Processing Analysis/                                
├── 2. Biomedical Image Analysis/                                        
├── 3. Human Action Recognition/                                         
├── 4. Instrumentation and Processing for Biomedical Applications/       
├── 5. Visual Recognition/                                               
├── 6. Practicum - Facial recognition login system/                      
├── 7. Master's thesis - Ordinal Methods for Monocular Depth Estimation/ 
└── README.md                                                            
```

## 📚 **Module Summaries**

1️⃣ **Advance Image Processing Analysis**
Implementations of classical variational, probabilistic, and modern generative methodologies for digital image restoration and segmentation. Core techniques cover Total Variation (ROF) denoising, Perona-Malik anisotropic diffusion, Markov Random Fields via Graph Cuts, Active Contours, super-resolution reconstruction, and deep learning-based colorization.

2️⃣ **Biomedical Image Analysis**
Practical computer vision and deep learning pipelines applied to medical modalities, specifically OCT scans. Features a 126-feature extraction and selection pipeline (LBP, GLCM, Gabor, HOG+PCA with Random Forest) for Diabetic Macular Edema (DME) detection, alongside deep semantic segmentation benchmarks evaluating Attention U-Net and LinkNet architectures for pathological fluid region extraction.

3️⃣ **Human Action Recognition**
End-to-end deep learning frameworks spanning affective computing, video classification, and 3D structural reconstruction. Includes fine-tuned ResNet-50 v2 models handling severe class imbalance for 7-class facial emotion recognition, spatio-temporal sequence modeling comparing 2D CNN+RNN baselines against Video Transformers, and a literature review on 2D-to-3D skeletal pose estimation.

4️⃣ **Instrumentation and Processing for Biomedical Applications**
Numerical analysis and digital image processing tools engineered for non-invasive clinical diagnostics. Key implementations cover color-space transformations for quantifying bulbar conjunctiva redness levels and a modular Python pipeline for automated aortic cross-section border detection, strut candidate selection (Hough transforms), and structural distance estimations.

5️⃣ **Visual Recognition**
Video processing, spatial-temporal filtering, and visual object tracking algorithms. Covers FIR/IIR temporal filtering, MOG/KNN adaptive background modeling, dense optical flow evaluation (Lucas-Kanade, Farnebäck, RLOF) on the MPI Sintel benchmark, and real-time Multi-Object Tracking (MOT) using the SORT algorithm evaluated on MOT20.

6️⃣ **Practicum - Facial Recognition Login System**
Development of a complete biometric authentication system developed in collaboration with a company. Evaluates synthetic-to-real domain transfer by training deep embedding networks (DenseNet-121 backbone) on synthetic face data (DCFace) and evaluating identity verification performance on real face datasets (CASIA-WebFace) using standard Triplet Loss, Semi-Hard Triplet Mining, and ArcFace loss functions.

7️⃣ **Master's Thesis - Ordinal Methods for Monocular Depth Estimation**
Research repository introducing novel spatial regularization loss terms (1-Side, 2-Sides, Multi-Scale) to penalize non-adjacent ordinal bin predictions across neighboring pixels in Monocular Depth Estimation (MDE). Features DeepLabv3 with a ResNet-101 backbone under Spacing-Increasing Discretization (SID) and proposes OrdinalMapping, a probabilistic neural module that learns to map discrete class distributions into smooth continuous depth outputs.


## 🛠️ Setup & Requirements

### Prerequisites

* Python: 3.8+
* Recommended dependencies: opencv-python, numpy, matplotlib, scikit-image, scipy, torch / tensorflow


## 👤 Author & Work Attribution

* **Author**: Sofía J. Lucía
* **Program**: International Master in Computer Vision (IMCV)
* **Attribution**: All code, practical laboratory assignments, code implementations, practicum developments, and thesis research in this repository were individually completed and developed entirely by myself.
