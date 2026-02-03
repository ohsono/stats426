# Quantifying CNN Robustness to Weather Corruptions in Traffic Sign Recognition
**Course:** Stats 426 | **Team:** Ayah Halabi, Setara Nusratty, Mark Rahal, Hochan Son | **Timeline:** 4 weeks
## Objective
Deep CNNs achieve 99% accuracy on clean traffic sign datasets but degrade significantly under real-world conditions. We quantify this robustness gap by measuring ResNet-50's performance on synthetic weather corruptions (rain, fog, low brightness) to identify which conditions and sign types are most vulnerable.
## Methodology

**Dataset:** GTSRB (German Traffic Signs) and LISA (US Traffic Signs)
- **Scope:** 5 classes (Stop, Yield, Speed 30/50/70 km/h)
- **Resolution:** 224×224 RGB images
- **Split:** 80% train / 20% validation (from training set) + separate test set

**Model:** ResNet-50 (pretrained on ImageNet)
- Fine-tune on clean 5-class subset
- Standard hyperparameters: Adam (lr=1e-4), 15 epochs, early stopping
- Target: >90% clean validation accuracy

**Corruptions** (Albumentations library, applied to test set only):
1. **Rain:** Synthetic droplets, reduced contrast
2. **Fog:** Atmospheric scattering (fog_coef=0.5)
3. **Low Brightness:** 30% brightness reduction

**Evaluation Metrics:**
- **Degradation Score:** Accuracy_corrupted / Accuracy_clean
- **Confusion Matrices:** 5×5 matrices per corruption type
- **Vulnerability Index:** Per-class susceptibility ranking

## Expected Results

| Metric | Expected Range | Hypothesis |
|--------|---------------|------------|
| Clean accuracy | 92-98% | Baseline performance |
| Rain degradation | below 0.85 | Moderate impact |
| **Fog degradation** | **below 0.75** | **Most damaging** (loss of high-frequency details) |
| Brightness degradation | below 0.90 | Least impact |

**Key Prediction:** Speed limit signs (text-dependent) will be more vulnerable than Stop/Yield (shape-dependent).

## Success Criteria

**Minimum:** ResNet-50 trained (>85%), degradation scores for 3 corruptions, confusion matrices, 6-page report
**Ideal:** >92% clean accuracy, clear vulnerability patterns identified, publication-quality analysis

## References

1. Stallkamp et al. (2012). Man vs. computer: Benchmarking ML for traffic sign recognition. *Neural Networks*, 32, 323-332.
2. Hendrycks & Dietterich (2019). Benchmarking neural network robustness to common corruptions. *ICLR*.
3. Qiao, X. (2023). Traffic sign recognition based on CNN. *Procedia Computer Science*, 220, 107-114.