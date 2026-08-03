# 4. Instrumentation and Processing for Biomedical Application (IPBMA)

This directory contains code, documentation, and experimental results for the Instrumentation and Processing for Biomedical Application module.

## 📚 Module Overview
This module focuses on applying computer vision, digital image processing, and numerical analysis techniques to solve practical biomedical problems.

More details: [https://imcv.eu/guide/2023-2024/ipbma/](https://imcv.eu/guide/2023-2024/ipbma/)


## 🔬 Practical Use Cases

### 🔹 Case 1: Redness Level of Bulbar Conjunctiva
Quantify and analyze the level of redness in the bulbar conjunctiva using digital image processing techniques.

* **Case 1_Level of redness.ipynb**: Interactive Jupyter Notebook containing image loading, color-space conversion, vessel segmentation, and redness metric computation.
* **Case 1_Level of redness.pdf**: Detailed project report summarizing the methodology, clinical context, and output results.

### 🔹 Case 2: Aorta Analysis
Perform automated geometric, structural, and distance estimation analysis on intravascular/aortic cross-sectional images (e.g., detecting borders, identifying struts via Hough transforms, radial edge detection).

* **main.py**: Modular Python script containing utility functions for edge detection, distance estimation from circumference centers, vector rotations, and multi-order strut candidate selection.
* **Case 2_Aorta Analysis.ipynb**: Primary analytical notebook leveraging functions from main.py to evaluate the dataset and extract aortic parameters.
* **Case 2_Aorta Analysis.pdf**: Comprehensive document detailing the algorithm pipeline, mathematical formulations, and experimental outputs.
