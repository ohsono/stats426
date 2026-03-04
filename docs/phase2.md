# Phase_2_Architecture_Scaling.md

## 1. Overview
This phase defines the progression from a simple baseline to deployment-ready, state-of-the-art models. The architecture must balance high expressive power with low-latency constraints suitable for edge inference in vehicles.

## 2. Model Progression

### 2.1 Baseline CNN


[Image of Convolutional Neural Network architecture]

* **Structure:** 2-3 Convolutional layers followed by Max Pooling and a fully connected head.
* **Purpose:** Validates the data pipeline and establishes a floor for expected performance.

### 2.2 Advanced CNN (Spatial Invariance)
* **Structure:** Integration of a Spatial Transformer Network (STN) module before the primary convolutional blocks.
* **Purpose:** The STN learns affine transformations to auto-center and crop the region of interest internally. `BatchNorm2d` is heavily utilized to stabilize the internal covariate shift caused by mixing German (GTSRB) and US (LISA/BDD100K) color distributions.

### 2.3 ResNet10 (Edge-Deployable Workhorse)

* **Structure:** A modified ResNet10.
* **Modification:** The standard 7x7 stride-2 convolution in the initial stem destroys too much spatial information for 64x64 inputs. Replace it with a `3x3 stride-1` convolution. 
* **Purpose:** Achieves deep hierarchical feature extraction without vanishing gradients, maintaining a small enough footprint for ONNX/TensorRT compilation on edge hardware.

### 2.4 Vision Language Model (Orion)

* **Structure:** Pre-trained VLM adapted via Low-Rank Adaptation (LoRA).
* **Purpose:** Casts the classification as a Visual Question Answering (VQA) task ("What traffic sign is this?"). Excellent for zero-shot generalization on text-heavy signs (e.g., "HOV lane description").