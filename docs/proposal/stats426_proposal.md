# Traffic Sign Classification Under Complex Conditions: Beyond CNNs
**Course:** Stats 426  
**Team:** Ayah Halabi, Setara Nusratty, Mark Rahal, Hochan Son  
**Dataset:** Traffic Sign Dataset – Classification (Kaggle, 58 classes)

## Objective
Since AlexNet conquered image classification in 2012, deep learning has revolutionized computer vision. However, machines still struggle to understand images in harsh conditions—variable lighting, occlusion, weather effects, and motion blur remain significant challenges. Our objective is to learn current knowledge in traffic sign recognition and implement Vision-Language Models (VLMs) and sophisticated algorithms to achieve beyond 90% accuracy in complex real-world conditions, advancing beyond traditional CNN approaches.

## Problem Definition & Motivation
Traffic sign recognition (TSR) is critical for autonomous driving and intelligent transportation systems. While standard CNNs achieve high accuracy on clean datasets (97-99% as shown in Qiao et al., 2023), performance degrades significantly under real-world conditions. This project addresses three key questions:

1. **Robustness Gap:** How much does CNN performance degrade under adverse conditions (poor lighting, occlusion, weather), and can advanced architectures close this gap?
2. **Beyond Standard CNNs:** Can Vision-Language Models or hybrid approaches surpass traditional CNN accuracy by leveraging semantic understanding of traffic signs?
3. **Error Patterns:** What failure modes persist across different architectures, and what do they reveal about model limitations?

## Methodology

### Data Preparation & Augmentation
- **Dataset:** Kaggle Traffic Sign Dataset (58 classes, ~50,000 images, 32×32 RGB)
- **Preprocessing:** HSV-based color segmentation (following Qiao et al.) to isolate ROI, cropping redundant backgrounds, normalization
- **Augmentation:** Simulate harsh conditions with rotation, brightness/contrast variation, Gaussian blur, synthetic occlusion, and weather effects (rain, fog)
- **Split:** 70% train, 15% validation, 15% test (stratified to handle class imbalance)

### Model Architecture Strategy
1. **Baseline CNN:** Custom architecture with batch normalization and dropout (reproducing Qiao et al.'s approach)
2. **Transfer Learning:** Fine-tuned ResNet-50 or EfficientNet-B0 pretrained on ImageNet
3. **Advanced Approach:** Vision-Language Model integration:
   - Extract CLIP embeddings as feature representations
   - Hybrid pipeline: CNN visual features + VLM semantic features
   - Text-guided classification using sign descriptions

### Training & Evaluation
- **Loss:** Categorical cross-entropy with class weights to address imbalance
- **Optimizer:** Adam with learning rate scheduling
- **Regularization:** Dropout, batch normalization, early stopping
- **Metrics:** Overall accuracy, per-class F1-scores, confusion matrices, robustness metrics on augmented test sets

### Baseline Comparison
- **Traditional ML:** Logistic Regression and Random Forest on flattened pixels and HOG features
- **Analysis:** Compare feature learning (CNN/VLM) vs. handcrafted features

## Expected Contributions
1. **Empirical Analysis:** Quantify accuracy degradation under simulated adverse conditions across multiple architectures
2. **VLM Integration:** Demonstrate whether semantic understanding from vision-language models improves robustness beyond pixel-level pattern matching
3. **Error Analysis:** Systematic categorization of failure modes (sign similarity, environmental factors) with Grad-CAM visualizations

## Deliverables
- Trained models with performance benchmarks on clean and augmented test sets
- Comprehensive error analysis with visualizations (confusion matrices, Grad-CAM)
- Final report documenting methodology, results, and insights for advancing TSR robustness

**Target:** Achieve >90% accuracy on complex-condition test set, demonstrating measurable improvement over baseline CNN approaches.
