# Quantifying CNN Robustness to Weather Corruptions in Traffic Sign Recognition

**Course:** Stats 426 - Deep Learning
**Team:** Ayah Halabi, Setara Nusratty, Mark Rahal, Hochan Son
**Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)
**Timeline:** 4 weeks (estimated)

---

## 1. Abstract

Convolutional Neural Networks (CNNs) achieve near-perfect accuracy on clean traffic sign benchmarks (>99% on GTSRB), yet their performance in real-world conditions remains underexplored. This project quantifies the robustness gap by systematically measuring how ResNet-50's accuracy degrades under synthetic weather corruptions (rain, fog, and low brightness). Using a controlled subset of 5 traffic sign classes from GTSRB, we will generate corrupted test sets and analyze failure patterns through confusion matrices. Our goal is to provide empirical evidence of which weather conditions pose the greatest challenge to CNN-based Traffic Sign Recognition (TSR) systems and identify which sign types are most vulnerable to misclassification.

---

## 2. Introduction

### 2.1 Motivation

Autonomous vehicles and driver assistance systems rely on accurate traffic sign recognition to ensure safety. While modern CNNs achieve remarkable performance on curated datasets under ideal conditions, real-world deployment exposes models to adverse weather, variable lighting, and environmental occlusions. A model that achieves 99% accuracy on sunny test images may fail catastrophically during a foggy morning commute, posing serious safety risks.

Despite extensive research on TSR accuracy improvements, **robustness under distribution shift remains underexamined**. Most published benchmarks report performance on clean test sets, leaving practitioners uncertain about expected degradation in production environments.

### 2.2 Research Questions

This project addresses three core questions:

1. **Quantification:** How much does ResNet-50 accuracy degrade when traffic signs are subjected to rain, fog, and low-light conditions?
2. **Vulnerability Analysis:** Are certain sign types (e.g., speed limits vs. regulatory signs) more susceptible to weather-induced misclassification?
3. **Error Patterns:** What are the most common misclassifications under each weather condition, and do they reveal systematic model weaknesses?

### 2.3 Contributions

- **Empirical degradation benchmarks** for ResNet-50 on weather-corrupted traffic signs
- **Class-specific vulnerability analysis** identifying which signs fail under which conditions
- **Reproducible corruption pipeline** using Albumentations for future TSR robustness studies
- **Actionable insights** for improving model resilience in autonomous driving systems

---

## 3. Related Work

### 3.1 Traffic Sign Recognition

Traffic sign recognition has evolved from traditional computer vision approaches (HOG features + SVM) to deep learning methods. Pioneering work by Sermanet & LeCun (2011) applied CNNs to GTSRB, achieving 99.17% accuracy. Qiao et al. (2023) further demonstrated 99.9% accuracy using deeper architectures with HSV preprocessing and data augmentation.

However, these studies focus on **maximizing clean-set accuracy** rather than evaluating robustness to environmental shifts. The gap between laboratory performance and real-world reliability remains a critical challenge for deployment.

### 3.2 Robustness to Corruptions

Hendrycks & Dietterich (2019) introduced ImageNet-C, a benchmark for measuring CNN robustness to 15 common corruptions (blur, noise, weather). They found that standard CNNs suffer 20-40% accuracy drops under moderate corruption, revealing brittleness despite high clean accuracy. Similarly, Michaelis et al. (2019) showed that object recognition models trained on clean data fail when objects are partially occluded.

**Gap in literature:** While ImageNet-C evaluates general object recognition, traffic sign recognition has unique requirements—signs must be recognized at a distance, often in suboptimal conditions. GTSRB lacks a standardized corruption benchmark for weather-specific degradation analysis.

### 3.3 Domain Adaptation Approaches

Recent work explores domain adaptation to improve robustness. Stallkamp et al. (2012) proposed synthetic data augmentation during training to simulate real-world variability. Sun et al. (2020) used adversarial training to harden models against perturbations. However, these methods require retraining or access to corrupted training data, which may not be available.

Our work complements this literature by **quantifying the robustness gap of standard fine-tuned models** without additional hardening techniques, establishing a baseline for future improvement efforts.

---

## 4. Methodology

### 4.1 Dataset: GTSRB or LISA

**Source:** Stallkamp et al., "Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition" (2012)

**Dataset characteristics:**
- ~51,000 images across 43 traffic sign classes
- Pre-cropped and resized to variable resolutions (rescaled to 224×224 for ResNet-50)
- Real-world images captured under varying conditions (lighting, weather, viewing angles)
- Standard split: ~39,000 training, ~12,000 test images

**Class selection:**
To ensure statistical significance and manageable scope, we focus on **5 representative classes**:

| Class ID | Sign Type | Training Samples | Rationale |
|----------|-----------|------------------|-----------|
| 1 | Speed limit (30 km/h) | ~2,000 | Common speed limit |
| 2 | Speed limit (50 km/h) | ~2,100 | Most frequent class |
| 4 | Speed limit (70 km/h) | ~1,800 | Similar to 30/50, tests text robustness |
| 13 | Yield | ~2,000 | Triangular shape, distinct color |
| 14 | Stop | ~750 | Octagonal shape, critical for safety |

**Justification for 5 classes:**
- Includes both **text-dependent** (speed limits) and **shape-dependent** (Stop, Yield) signs
- Sufficient samples per class (>750) for robust statistics
- Covers different geometric shapes (circle, triangle, octagon)
- Reduces training time to <2 hours, enabling rapid iteration

### 4.2 Model Architecture: ResNet-50

**Selection rationale:**
- ResNet-50 is a standard baseline for image classification
- Pretrained on ImageNet (1.2M images), enabling transfer learning
- Well-documented, reproducible, available in PyTorch/torchvision
- Balances depth (50 layers) with computational efficiency

**No hyperparameter search:** We use standard transfer learning configurations to prioritize robustness evaluation over accuracy maximization.

**Validation strategy:** 80/20 train-validation split within the 5-class training set for early stopping.

### 4.3 Synthetic Corruption Pipeline

We simulate three common adverse conditions using the Albumentations library

### 4.4 Evaluation Metrics

#### 4.4.1 Accuracy Metrics
- **Overall accuracy:** $\text{Acc} = \frac{\text{Correct predictions}}{\text{Total predictions}}$
- **Per-class accuracy:** Accuracy calculated separately for each of the 5 classes

#### 4.4.2 Degradation Score
$$\text{Degradation Score} = \frac{\text{Accuracy}_{\text{corrupted}}}{\text{Accuracy}_{\text{clean}}}$$
A score of 1.0 indicates no degradation; lower scores indicate greater vulnerability.

#### 4.4.3 Confusion Matrices
5×5 confusion matrices for each condition (clean, rain, fog, brightness) to identify:
- Most common misclassifications
- Whether errors are systematic (e.g., 30 km/h → 50 km/h) or random

#### 4.4.4 Vulnerability Index
For each sign class $c$:
$$\text{Vulnerability}_c = 1 - \min(\text{Degradation}_{\text{rain}}, \text{Degradation}_{\text{fog}}, \text{Degradation}_{\text{brightness}})$$

Higher values indicate greater susceptibility to weather corruptions.

## 5. Experimental Design

### 5.1 Training Phase (Week 1)

**Objective:** Train ResNet-50 on clean 5-class subset to achieve >90% validation accuracy

**Steps:**
1. Filter GTSRB or (LISA) dataset to 5 selected classes
2. Apply minimal augmentation during training (random horizontal flip only)
3. Train for up to 15 epochs with early stopping (validation loss plateau)
4. Save best checkpoint based on validation accuracy

**Success criterion:** Validation accuracy >90% on clean images

### 5.2 Corruption Generation (Week 2)

**Objective:** Create 3 corrupted test sets and evaluate model

**Steps:**
1. Load clean test set (5 classes, preferred 2,400 images)
2. Apply each corruption transform to create 3 new test sets:
   - `test_rain/`: Rain corruption applied to all images
   - `test_fog/`: Fog corruption applied to all images
   - `test_brightness/`: Brightness corruption applied to all images
3. Run trained ResNet-50 inference on all 4 test sets (clean + 3 corrupted)
4. Record predictions and ground truth labels

**Success criterion:** All test sets generated, accuracy measured for each

### 5.3 Analysis Phase (Week 3)

**Objective:** Analyze degradation patterns and identify vulnerabilities

**Steps:**
1. Calculate overall and per-class degradation scores
2. Generate confusion matrices for each corruption type
3. Identify most common errors (e.g., "Speed 30 → Speed 50" under fog)
4. Rank sign classes by vulnerability index
5. Create visualizations (bar charts, heatmaps)

**Success criterion:** Clear narrative about which signs fail under which conditions

### 5.4 Documentation Phase (Week 4)

**Objective:** Write final report and prepare code repository

**Deliverables:**
- 6-8 page research report
- GitHub repository with training/evaluation scripts
- README with reproduction instructions

---

## 6. Expected Results

### 6.1 Hypothesis

We hypothesize that:
1. **Fog will cause the greatest degradation** (60-70% degradation score) due to loss of high-frequency details
2. **Speed limit signs will be more vulnerable** than Stop/Yield due to reliance on small text rather than distinct shapes
3. **Common error pattern:** Speed limit signs will be confused with each other (30 ↔ 50 ↔ 70)

### 6.2 Anticipated Outcomes

| Metric | Expected Range |
|--------|---------------|
| Clean accuracy | 92-98% |
| Rain degradation score | 0.75-0.85 |
| Fog degradation score | 0.65-0.75 |
| Brightness degradation score | 0.80-0.90 |

**Most vulnerable class:** Speed limit signs (text-dependent)
**Most robust class:** Stop sign (distinctive octagonal shape + red color)

## 7. Division of Labor

| Team Member | Primary Responsibility | Secondary Support |
|-------------|------------------------|-------------------|
| **Ayah Halabi** | Evaluation pipeline, confusion matrix analysis | Training script debugging |
| **Setara Nusratty** | Corruption implementation, visualization | Data preprocessing |
| **Mark Rahal** | Augmentation testing, degradation metrics | Report figures/tables |
| **Hochan Son** | ResNet-50 training, lead report writer | Statistical analysis |

**Collaboration model:**
- Week 1: Entire team works together on dataset and training (pair programming)
- Weeks 2-4: Parallel workstreams with daily async standups
- Week 4: All members contribute to report review

## 8. Resources & Requirements

### 8.1 Computational Resources
- **GPU:** Google Colab Pro ($10/month) or university cluster access
  - Free Colab tier may suffice (5 classes, 15 epochs ≈ 2 hours training)
- **Storage:** ~5 GB for GTSRB dataset + corrupted test sets
- **RAM:** 16 GB recommended for data loading

### 8.2 Software Stack
- **Framework:** PyTorch 2.8+, torchvision
- **Augmentation:** Albumentations 1.3+
- **Visualization:** matplotlib, seaborn
- **Utilities:** numpy, pandas, scikit-learn

### 9.3 Dataset Access
- GTSRB available via Kaggle or LISA[https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset]
- No licensing restrictions for academic use

## 10. Success Criteria

### Minimum Viable Outcome (Required)
- ✅ ResNet-50 trained with >85% clean accuracy on 5-class subset
- ✅ Degradation scores measured for all 3 corruption types
- ✅ Confusion matrices generated showing error patterns
- ✅ 6-page report documenting methodology and findings

### Ideal Outcome (Aspirational)
- ✅ Clean accuracy >92%
- ✅ Clear statistical separation between corruption types (e.g., fog worse than rain by >10%)
- ✅ Actionable insights for improving robustness (e.g., "augment training with fog")
- ✅ Publication-quality figures suitable for conference submission

## 11. References

1. Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks*, 32, 323-332.

2. Sermanet, P., & LeCun, Y. (2011). Traffic sign recognition with multi-scale convolutional networks. *Proceedings of IJCNN*, 2809-2813.

3. Qiao, X. (2023). Research on traffic sign recognition based on CNN deep learning network. *Procedia Computer Science*, 220, 107-114.

4. Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *Proceedings of ICLR*.

5. Michaelis, C., et al. (2019). Benchmarking robustness in object detection: Autonomous driving when winter is coming. *arXiv preprint arXiv:1907.07484*.

6. Sun, Y., et al. (2020). Circle loss: A unified perspective of pair similarity optimization. *Proceedings of CVPR*, 6398-6407.

7. Buslaev, A., et al. (2020). Albumentations: Fast and flexible image augmentations. *Information*, 11(2), 125.

8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of CVPR*, 770-778.
