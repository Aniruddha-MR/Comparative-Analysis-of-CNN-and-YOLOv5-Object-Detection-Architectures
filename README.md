# Comparative Analysis of CNN and YOLOv5 Object Detection Models

This repository contains the report and findings from a comprehensive study comparing the performance of Convolutional Neural Networks (CNN) and YOLOv5 for object detection. The research explores their architecture, feature extraction mechanisms, and experimental performance to provide insights into their respective strengths and limitations.

## Project Overview
The primary objectives of this project are:
- To investigate the performance of CNN and YOLOv5 architectures.
- To understand the impact of architectural modifications and hyperparameter tuning on CNN.
- To evaluate YOLOv5’s object detection capabilities and generalization across different datasets.
- To provide a comparative analysis of computational efficiency and predictive performance.

## Methodology
### Dataset
- **CNN Model:** Trained on the CIFAR-10 dataset (60,000 images across 10 classes).
- **YOLOv5 Model:** Evaluated on a custom 5-image dataset and pre-trained on the COCO dataset.

### Preprocessing
- Normalization, one-hot encoding for CNN.
- Data augmentation for CNN using random rotations, shifts, shears, zooms, and flips.

### CNN Architecture
- Multi-layer convolutional network with ReLU activation, max-pooling, and dropout (50%).
- Explored different configurations of stride and padding to study their impact on performance.
- Hyperparameter tuning using Keras Tuner for optimized learning rate, dropout rate, and dense layer size.

### YOLOv5 Implementation
- Pre-trained YOLOv5s model with confidence and IoU thresholds set at 0.25 and 0.45, respectively.
- Evaluated for inference speed and cross-domain generalization capabilities.

## Results Summary
### CNN Performance
- **Baseline Accuracy:** 77.02% on the CIFAR-10 test set.
- **Hyperparameter Tuning:** Improved validation accuracy to 78.70%.
- **Stride and Padding Analysis:** Significant performance variations with different configurations:
  - Stride 1 with 'Valid' padding achieved 74.06% accuracy.
  - Stride 2 with 'Same' padding resulted in lower accuracy (66.51%).

### YOLOv5 Performance
- **High Object Detection Accuracy:** Excellent performance on images from the COCO dataset.
- **Inference Speed:** Fast inference times suitable for real-time applications.
- **Cross-Domain Limitation:** Decreased accuracy and low confidence scores on out-of-distribution images.

## Comparative Analysis
| Aspect                   | CNN                    | YOLOv5            |
|--------------------------|------------------------|-------------------|
| **Data Complexity**       | CIFAR-10               | Custom Dataset    |
| **Feature Extraction**    | Detailed, Layer-by-Layer | Grid-based, Holistic |
| **Computational Efficiency** | Moderate              | High-Speed Inference |
| **Generalization**        | Domain-Specific        | Limited Cross-Domain |

## Conclusion
The study highlights the unique trade-offs between CNN and YOLOv5. While CNN provides better control over feature extraction and generalizes well within its domain, YOLOv5 excels in real-time object detection with high computational efficiency but struggles with cross-domain generalization. Choosing the right model depends on the specific requirements of the task, such as accuracy, speed, or domain flexibility.
