# Phase_4_Trustworthy_Evaluation.md

## 1. Overview
In safety-critical autonomous systems, standard accuracy metrics (like Top-1 Accuracy) are dangerously insufficient. A model must not only be accurate but also highly reliable and self-aware of its uncertainty. Trustworthy AI principles dictate rigorous Out-of-Distribution (OOD) testing and calibration tracking.

## 2. Core Trustworthy Metrics

### 2.1 Expected Calibration Error (ECE) & Reliability Diagrams

* **Definition:** The difference between the model's confidence (softmax probability) and its actual accuracy. 
* **Requirement:** If the ResNet10 outputs a 99% confidence that a sign is "Speed Limit 65", it must be correct 99% of the time. Overconfident misclassifications on stop signs or yield signs are unacceptable. Apply Temperature Scaling post-training if the model is uncalibrated.

### 2.2 Per-Class Precision, Recall, and F1-Score
* **Implementation:** Generate a detailed classification report using `scikit-learn`. 
* **Focus:** Pay close attention to the recall on critical signs (Stop, Yield, Do Not Enter). A false positive is an annoyance; a false negative is a system failure.

### 2.3 Out-of-Distribution (OOD) Degradation Testing
* **Execution:** Evaluate the finalized model against the curated 10% Challenge Split (e.g., exclusively night-time BDD100K images, heavily occluded signs, or extreme weather conditions).
* **Metric:** Measure the "generalization gap"—the absolute drop in F1-score between the In-Domain Test set and the OOD Challenge set. A robust model will keep this gap narrow.