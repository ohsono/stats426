# Lecture 3: Convolutional Neural Networks (CNNs) - Summary

**Course:** STAT 426
**Instructor:** George Michailidis (gmichail@ucla.edu)

---

## Overview
This lecture introduces Convolutional Neural Networks (CNNs), which are specifically designed for processing data with spatial structure like images. CNNs address the limitations of Multi-Layer Perceptrons (MLPs) when dealing with image data.

---

## Key Concepts

### 1. **Why MLPs Fail for Images**
- **Flattening destroys spatial structure**: Converting 2D images to 1D vectors loses geometric relationships between pixels
- **Not translation invariant**: MLPs treat the same object in different positions as completely different inputs
- **Parameter explosion**: A 200×200 RGB image with 1,000 hidden units requires ~120 million parameters in the first layer alone
- **Permutation test failure**: MLPs perform identically on permuted (scrambled) images, unlike human vision

### 2. **Core Principles of CNNs**

#### **Translation Invariance (Weight Sharing)**
- Same filter is applied across all spatial locations
- If a vertical edge detector is useful at position (10,10), it's useful at (50,50)
- Drastically reduces parameters while maintaining expressiveness

#### **Locality (Sparse Connectivity)**
- Each neuron only connects to a small local region of input
- Exploits the statistical prior that nearby pixels are highly correlated
- Distant pixels have weak correlations and don't need direct connections

---

## 3. **Convolution Operation**

### Mathematical Definition
For 2D input and kernel:
```
Output(i,j) = Σ Σ Input(i+a, j+b) × Kernel(a,b)
            a b
```

### Key Properties
- **Cross-correlation vs Convolution**: Deep learning uses cross-correlation (no kernel flip) but calls it "convolution"
- **Filter as template matcher**: High activation when input pattern matches the filter
- **Dot product interpretation**: `w·x = ||w||||x||cos(θ)` - maximum when angle is small

### Multi-Channel Convolution
- **Input**: H × W × C_in (e.g., RGB image: H × W × 3)
- **Kernel**: K × K × C_in
- **Process**: Convolve each channel independently, then sum results
- **Output**: Single 2D feature map (or multiple with C_out filters)

---

## 4. **Padding and Stride**

### Output Size Formula
```
O = ⌊(W - K + 2P) / S⌋ + 1
```
Where:
- W = input width/height
- K = kernel size
- P = padding
- S = stride

### Common Configurations
- **Valid Convolution**: P=0 (output shrinks)
- **Same Convolution**: P=(K-1)/2 with S=1 (output size = input size)
- **Strided Convolution**: S>1 (downsampling)

---

## 5. **Pooling Layers**

### Purpose
- **Dimensionality reduction**: Reduces spatial resolution (e.g., 224×224 → 112×112)
- **Translation invariance**: Small shifts in input don't change output
- **No parameters**: Fixed operation (no learning)

### Types
- **Max Pooling** (most common): Takes maximum value in each window
  - Intuition: "Did this feature occur anywhere in this region?"
  - Standard: 2×2 window, stride 2 (halves dimensions)

- **Average Pooling**: Takes mean of values in window
  - Less common in intermediate layers
  - Used as Global Average Pooling (GAP) at network end

### Benefits
- 75% data reduction with 2×2 pooling
- Forces network to keep only salient features
- Provides robustness to small translations

---

## 6. **Receptive Fields**

### Definition
The receptive field of a neuron is the region of the original input image that influences that neuron's activation.

### Growth Through Layers
- **Layer 1**: Receptive field = kernel size (e.g., 3×3)
- **Layer 2**: Each neuron sees a 3×3 patch of Layer 1, which itself saw 3×3 patches → effective 5×5 receptive field
- **Rule of thumb**: For 3×3 kernels with stride 1: RF ≈ 1 + 2×(depth)

### Feature Hierarchy
- **Early layers**: Edges, gradients, colors
- **Middle layers**: Textures, corners, circles
- **Deep layers**: Object parts (eyes, wheels, text)
- Network learns this hierarchy automatically through backpropagation

---

## 7. **Channel Operations**

### Multiple Output Channels
- Need multiple filters to detect different patterns simultaneously
- **Filter bank**: C_out distinct kernels, each producing one feature map
- **Parameters**: (K² × C_in + 1) × C_out

### 1×1 Convolutions
**Three main purposes:**

1. **Bottleneck (efficiency)**: Compress channels (e.g., 512 → 64) before expensive operations
2. **Feature fusion**: Combine information across channels at each pixel
   - Acts as MLP applied independently to each pixel
   - Can learn combinations like "red vertical edge" from "red" and "vertical" channels
3. **Adding depth**: Increases non-linearity without reducing spatial resolution

**Mathematical view**:
- Linear combination of input channels: `Output = w₁·Ch₁ + w₂·Ch₂ + ... + wₙ·Chₙ`
- No spatial mixing (pixels don't interact)
- Only channel mixing (colors/features combine)

---

## 8. **Data Augmentation**

### The Problem
- CNNs have translation equivariance but lack rotation and scale invariance
- A vertical edge detector cannot detect horizontal edges
- Network must see variations during training to handle them

### Common Techniques

**Geometric:**
- Horizontal/vertical flipping
- Random rotation (±15°)
- Random cropping/zooming
- Translation shifts

**Photometric:**
- Color jitter (brightness, contrast, saturation)
- Gaussian noise injection
- Cutout (random masking)

### Benefits
- Acts as powerful regularizer
- Prevents overfitting
- Increases effective dataset size
- Forces network to learn invariant features

---

## 9. **CNN Architecture Pipeline**

### Standard Structure

1. **STEM (Rapid Compression)**
   - Stride-2 convolution
   - Quickly reduces spatial dimensions
   - Extracts raw features

2. **BODY (Deep Feature Extraction)**
   - Repeated Conv-BN-ReLU blocks
   - **Batch Normalization**: Re-centers activations to stabilize training
   - Builds hierarchical feature representations
   - May include multiple stages with pooling

3. **HEAD (Classification)**
   - Global Average Pooling: 512×7×7 → 512×1×1
   - Linear layer: 512 → num_classes (e.g., 10)
   - Softmax: Converts logits to probabilities

---

## 10. **Biological Inspiration**

### Hubel & Wiesel (1959)
Discovered two types of cells in cat visual cortex:

- **Simple Cells**: Respond to edges at specific orientations and locations
  - Analogous to convolutional filters

- **Complex Cells**: Respond to edges anywhere in a region (motion/shift invariant)
  - Analogous to pooling layers

### Neocognitron (Fukushima, 1980)
- First architecture with S-layers (simple/conv) and C-layers (complex/pooling)
- Lacked backpropagation (used unsupervised Hebbian learning)
- Paved the way for modern CNNs

---

## Key Formulas

### Convolution Output Size
```
Output_height = ⌊(H - K_h + 2P) / S⌋ + 1
Output_width  = ⌊(W - K_w + 2P) / S⌋ + 1
```

### Parameter Count
```
Params = (K² × C_in + 1) × C_out
```
Where +1 accounts for bias term

### Softmax (Classification)
```
P(Class_i) = e^(logit_i) / Σ_j e^(logit_j)
```

---

## Important Design Choices

### Kernel Size
- **3×3**: Most common (VGGNet, ResNet)
- **5×5, 7×7**: Larger receptive field but more parameters
- **1×1**: Channel mixing, bottleneck layers

### Padding
- **Same**: Preserves spatial dimensions (easier architecture design)
- **Valid**: No padding (dimensions shrink)

### Stride
- **1**: Standard (dense sampling)
- **2**: Downsampling (alternative to pooling)

---

## Advantages of CNNs over MLPs

1. **Parameter efficiency**: 10⁸ → 10⁵ parameters through weight sharing
2. **Translation equivariance**: Features detected regardless of position
3. **Hierarchical learning**: Automatic feature hierarchy from simple to complex
4. **Spatial structure preservation**: Works directly on 2D/3D data
5. **Inductive bias**: Encodes assumptions about locality and stationarity

---

## Limitations

1. **Not rotation invariant**: Requires data augmentation
2. **Not scale invariant**: Must see different scales during training
3. **Limited geometric transformations**: Cannot handle complex deformations
4. **Require large datasets**: Need many examples to learn robust features
5. **Black box**: Difficult to interpret learned features

---

## Conclusion

CNNs revolutionized computer vision by encoding spatial inductive biases through:
- **Local connectivity** (locality)
- **Weight sharing** (translation equivariance)
- **Pooling** (translation invariance)
- **Hierarchical architecture** (feature composition)

These design principles enable CNNs to learn powerful visual representations efficiently, making them the foundation of modern computer vision systems.

---

## Next Steps
- Study specific architectures: LeNet, AlexNet, VGGNet, ResNet
- Understand advanced concepts: skip connections, attention mechanisms
- Explore applications: object detection, segmentation, generation
