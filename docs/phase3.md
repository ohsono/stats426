# Phase_3_Cross_Domain_Training.md

## 1. Overview
Mixing clean datasets (GTSRB) with noisy, real-world datasets (BDD100K) immediately usually results in catastrophic forgetting or suboptimal local minima. A structured Curriculum Learning approach is required. It is critical to track all experiment configurations and custom training loops in your private Git repository to maintain reproducibility.

## 2. Curriculum Schedule

### Stage 1: Fundamental Geometric Learning (Epochs 1-20)
* **Data:** 100% GTSRB + LISA.
* **Objective:** Train the model to recognize core shapes (octagons, inverted triangles, circles) and high-contrast symbols.
* **Hyperparameters:** Higher learning rate (e.g., `1e-3` with AdamW).

### Stage 2: Real-World Adaptation (Epochs 21-40)
* **Data:** Introduce BDD100K crops. 
* **Objective:** Force the network to recognize the learned geometries under motion blur, low light, and occlusion.
* **Hyperparameters:** Unfreeze all layers (if using a pre-trained backbone) and drop the learning rate using a `CosineAnnealingLR` scheduler to gently adjust the weights.

### Stage 3: Domain Adversarial Training (Optional / Advanced)

* **Implementation:** If the model struggles to generalize to BDD100K, attach a Gradient Reversal Layer (GRL) and a secondary domain-classifier head.
* **Objective:** The domain head tries to predict if the image came from GTSRB, LISA, or BDD100K. The GRL reverses these gradients to the feature extractor, forcing the convolutional layers to learn *domain-invariant* features.