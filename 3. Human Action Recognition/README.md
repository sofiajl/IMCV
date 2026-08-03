# Human Action Recognition (HAR)

This directory contains lab assignments, deep learning implementations, and research summaries covering Facial Emotion Recognition, Temporal Action Classification (CNN+RNN & Video Transformers), and 3D Human Pose Estimation literature review.


## 📚 Module Overview

Core Topics: Facial Affective Computing, Spatio-Temporal Sequence Classification, Recurrent Networks, Video Transformers, and Single-Image 3D Skeleton Reconstruction.

More details: https://imcv.eu/guide/2023-2024/har/

## 🧪 Laboratory Assignments
### 🎭 Lab 2: Facial Emotion Recognition

Focuses on evaluating and fine-tuning ResNet-50 v2 backbones for 7-class facial expression classification under real-world dataset constraints, including severe class imbalance, data augmentation strategies, loss weighting schemes, and over/under-sampling techniques.

* **01_data_analysis.ipynb**: Exploratory data analysis (EDA) covering pixel-level dataset inspection, 7-class distribution analysis, image sample visualization, and pre-processing pipeline setup.
* **02_1 to 02_6 ResNet50V2 Iterative Experiments**: Systematic model training across 6 experimental setups—testing data augmentation (`ImageDataGenerator`), macro/weighted metrics, class-weighting, baseline (no augmentation), and synthetic resampling.
* **03_pred_visualization.ipynb**: Comparative analysis across experiments, prediction evaluations, confusion matrices, and failure mode inspection.
* **Lab2_Emotion_Recognition.pdf**: Detailed report summarizing the ResNet-50 v2 pre-activation bottleneck architecture, hyperparameter choices, data augmentation specs, and comparative performance across all 6 experiments.

### 🎬 Lab 3: Human Action Recognition in Video

Compares deep spatial-temporal architectures for multi-frame action classification in video sequences.

* **CNN_RNN_classifier.ipynb**: Two-stage framework using pre-trained 2D CNNs (DenseNet121/InceptionV3) to extract frame features ($1024$-dim), followed by RNNs (LSTM/GRU) for temporal sequence modeling.
* **Transformer_based_classifier.ipynb**: Attention-based spatio-temporal modeling leveraging Video Transformers for long-range dependency capture across video frames.
* **Lab_3_Human_Action_Recognition.pdf**: Report covering video preprocessing, zero-padding ($60$ frames max), and performance evaluation across $6$ action classes.


### 🧍 Revision Work: 3D Human Pose Estimation
* **Paper sumary.pdf**: Literature review and technical summary on reconstructing 3D skeletal keypoints/human pose representations from a single 2D input image (lifting 2D joint detections to 3D geometry).
