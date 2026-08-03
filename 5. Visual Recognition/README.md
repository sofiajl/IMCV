# 5. Visual Recognition

This directory contains my individual implementations and experimental notebooks for the laboratory cases in the **Visual Recognition** module.

## 📚 Module Overview

This module focuses on video processing, spatial-temporal filtering, motion detection, optical flow estimation, and object tracking algorithms.

More details: https://imcv.eu/guide/2023-2024/vr/

## 🧪 Practical Labs & Project Overview

### 🔹 Lab 1: Video Sampling & Filtering
Processing video frame sequences through temporal digital signal filtering techniques for video enhancement and edge extraction.  


* **FIR and IIR Filters.ipynb**: Implementation of Finite and Infinite Impulse Response filters for temporal video smoothing and differentiation.
* **VR - Digital filters.pdf**: Detailed analysis comparing reconstruction quality (PSNR/SSIM) and computational efficiency across filter types.  

### 🔹 Lab 2: Motion Detection
Identifying moving regions in dynamic scenes using frame comparison and background modeling.

* **frame-diff.ipynb**: Fast temporal differencing combined with morphological filtering for moving object detection.  
* **background-extraction.ipynb**: Adaptive background modeling using Gaussian Mixture Models (MOG) and K-Nearest Neighbors (KNN).  
* **VR - Motion Detection.pdf**: Lab report evaluating precision, IoU, and performance tradeoffs across foreground segmentation techniques.  


### 🔹 Lab 3: Optical Flow
Estimating dense or sparse motion vectors across consecutive video frames.

* **optical-flow.ipynb**: Implementation of classical optical flow estimation algorithms (e.g., Lucas-Kanade, Farnebäck, Robust Local Optical Flow (RLOF)).
* **VR - Optical flow.pdf**: Experimental parameter tuning report and quantitative error evaluation (MAE, EPE, MRSE) on the MPI Sintel dataset.  



### 🔹 Lab 4: Moving Detection & Tracking

Estimating the state trajectory and bounding box location of an isolated target across a video sequence.  

* **kalman-filter.ipynb**: Target tracking using Kalman Filtering for motion estimation and noise reduction.
* **VR - Tracking.pdf**: Comparative evaluation of state-space modeling vs. histogram tracking on the VOT benchmark.



### 🔹 Final Project: Video and Motion Analysis
Real-time Multi-Object Tracking (MOT) using the SORT (Simple Online and Realtime Tracking) algorithm framework.

* **sort.py**: Core Python module implementing the SORT algorithm (Kalman Filter + Hungarian algorithm for data association).
* **SORT.ipynb**: Notebook demonstrating target tracking, evaluation metrics, and visual pipeline outputs.
* **SORT.pdf**: Original paper Simple Online and Realtime Tracking (by Alex Bewley , Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft)
* **VR_Project_on_Video_and_Motion_Analysis.pdf**: Comprehensive project report evaluating tracking performance (MOTA, MOTP, ID Switches) on the MOT20 benchmark.
