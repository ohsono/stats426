Phase 1: Data Harmonization & The 70-10-10-10 Strategy
To train seamlessly across three distinct datasets, you need a unified ontology and a highly tuned PyTorch DataLoader strategy.

1. The 70-10-10-10 Split Strategy
Instead of a standard Train/Val/Test split, allocate the final 10% to an explicitly curated Challenge/OOD Set.

70% Train: Core feature learning.

10% Validation: Hyperparameter tuning, early stopping, and learning rate scheduling.

10% In-Domain Test: Standard benchmarking (images drawn from the same distribution as the training set).

10% OOD / Challenge Test: Hand-picked hard examples (e.g., night-time BDD100K frames, heavy rain, or heavily occluded LISA signs) to measure true real-world robustness and calibration.

2. DataLoader Tuning per Dataset
You will need a unified PyTorch Dataset wrapper that standardizes the inputs. * GTSRB (The Clean Source): Currently 30x30. Upscale everything to 64x64 or 112x112 to leave room for spatial dimensions as you move to ResNet/VLM.

LISA (The Domain Bridge): US signs, clean crops. Map the LISA string labels to your unified integer index. Use torchvision.transforms for standard augmentation (RandomAffine, ColorJitter).

BDD100K (The Real-World Target): Extract crops using the bounding box JSONs. Because BDD100K suffers from heavy class imbalance and motion blur, wrap this dataset in a WeightedRandomSampler to oversample minority classes (like rare construction signs). Apply Gaussian blur and heavy contrast transformations to simulate dashcam artifacts.

Phase 2: Progressive Architecture Scaling
1. Baseline CNN (Completed in Stage A)

Purpose: Sanity check. Proves that 2 Conv layers + 1 FC layer can extract basic geometric edges and color blobs.

2. Advanced CNN

Purpose: Translation and scale invariance.

Modifications: Add Batch Normalization to stabilize training across the different color histograms of GTSRB vs. LISA. Add a Spatial Transformer Network (STN) module at the input. The STN learns to automatically center, rotate, and zoom in on the traffic sign before the convolutions process it, which is vital for the weird angles in BDD100K.

3. ResNet10 (The Edge-Deployable Workhorse)

Purpose: Learn deep hierarchical features without vanishing gradients. ResNet10 is ideal because it is lightweight enough for real-time vehicular inference (TensorRT/ONNX) but highly expressive.

Modifications: Replace the standard 7x7 stride-2 convolution in the ResNet stem with a 3x3 stride-1 convolution. Standard ResNets aggressively downsample the image (designed for 224x224 ImageNet). If you feed a 64x64 cropped sign into a standard ResNet, the spatial resolution collapses too early.

4. VLM (Orion - Optional, but Cutting-Edge)

Purpose: Zero-shot generalization and handling long-tail, text-heavy signs (e.g., "HOV lane description").

Modifications: Treat classification as a Visual Question Answering (VQA) task. Prompt the model with: "What traffic sign is visible in the center of this crop?" Fine-tune using Low-Rank Adaptation (LoRA) to adapt the VLM to your specific label nomenclature without destroying its pre-trained world knowledge.

Phase 3: The Cross-Domain Training Method
Do not dump all three datasets into a single pool immediately. Use Curriculum Learning.

Epochs 1-20 (Source Pre-training): Train the model exclusively on GTSRB and LISA. The network learns the fundamental geometries (octagons, triangles, circles) and colors in high resolution.

Epochs 21-40 (Domain Adaptation): Introduce the BDD100K crops. Unfreeze the lower layers but drop the learning rate (e.g., CosineAnnealingLR).

Advanced Tactic: If performance drops on BDD100K, implement a Domain Adversarial Neural Network (DANN) approach. Add a domain classifier head that tries to guess if an image is from GTSRB, LISA, or BDD100K. Reverse the gradients from this head during backprop so the core feature extractor learns representations that are domain-invariant.

Phase 4: Trustworthy Evaluation
Relying strictly on accuracy is insufficient for safety-critical AI systems like traffic sign recognition.

Per-Class Precision/Recall: A false positive on a "Yield" sign is annoying; a false positive on a "Speed Limit 80" when it's actually a "Stop" sign is fatal.

Expected Calibration Error (ECE): Plot a reliability diagram. If your ResNet10 outputs a 99% softmax confidence that a sign is "No Left Turn," it should be correct 99% of the time. Uncalibrated models are dangerous in autonomous systems.

OOD Degradation Testing: Evaluate the model specifically on the final 10% Challenge Split. Measure how much the accuracy drops from the In-Domain Test to the OOD Test. A resilient model should keep this "generalization gap" to a minimum.