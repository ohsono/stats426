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

**Translation invariance** means the model produces the **same output regardless of where in the input a feature appears**.

```
Cat top-left:     Cat center:       Cat bottom-right:
🐱 . . . . .     . . . . . .       . . . . . .
. . . . . .       . . 🐱 . . .     . . . . . .
. . . . . .       . . . . . .       . . . . . 🐱

All three → same output: "cat"
```

**Why MLPs fail at this:**
MLPs assign fixed, separate weights to each pixel position. The same cat shifted by one pixel activates completely different weights — the model must relearn the cat for every possible position.

**How CNNs achieve it:**
- **Shared weights (convolution)**: The same filter acts as a template scanning across all spatial locations (if a vertical edge detector is useful at position 10,10, it's useful at 50,50).
- **Pooling layers**: Summarize regions so small shifts don't change the output.
- Drastically reduces parameters while maintaining expressiveness.

**Translation Invariance vs. Equivariance:**

| Concept | Meaning | Example | CNN Component |
|---|---|---|---|
| **Equivariance** | Output shifts exactly mapping how the input shifts | "Where is the cat?" → Location shifts | Convolution layers |
| **Invariance** | Output stays exactly the same when input shifts | "Is there a cat?" → Always yes | Pooling / Global Avg Pooling |

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

### Definition in Plain English

The **receptive field** is simply the **"field of view"** of a single neuron. It answers the question: *"How much of the original image can this specific neuron actually 'see'?"*

> 🎯 **Analogy:** Imagine looking at a giant mural on a wall.
> - **Layer 1** is like looking through a tiny **drinking straw** — you can only see an area 3 pixels wide.
> - **Deep layers** are like stepping back and looking through a **large window** — you can see almost the entire mural.

Even if you go $50$ layers deep, a neuron in layer $50$ is still just a single number (acting like a single pixel). But that one number was calculated using information from a massive patch of the original image — that patch is its receptive field.


### Growth Through Layers (How $RF \approx 1 + 2 \times \text{depth}$ works)

Every time you add a $3 \times 3$ convolutional layer (with stride 1), the receptive field grows by **2 pixels** in each dimension (1 pixel on the left/top, 1 pixel on the right/bottom).

**Visual Breakdown:**

Let's trace what a single output pixel depends on back through the layers:

**Depth 1 (1st Conv Layer):**
- 1 output pixel looks at a **$3 \times 3$** patch of the original image.
- $RF = 3$   *(Formula: $1 + 2(1) = 3$)*
```
Image: [x x x]  ← 3 pixels
Output:  (o)    ← 1 pixel
```

**Depth 2 (2nd Conv Layer):**
- 1 output pixel looks at a $3 \times 3$ patch of Layer 1.
- But *each* of those Layer 1 pixels looks at a $3 \times 3$ patch of the image.
- The center pixel sees the middle 3. The left pixel sees 1 extra pixel to the left. The right pixel sees 1 extra pixel to the right.
- Total image pixels seen: $3 + 1 (\text{left}) + 1 (\text{right}) =$ **$5 \times 5$**
- $RF = 5$   *(Formula: $1 + 2(2) = 5$)*
```
Image:   [x x x x x]  ← 5 pixels
Layer 1:   [o o o]    ← 3 pixels
Layer 2:     (O)      ← 1 pixel
```

**Depth 3 (3rd Conv Layer):**
- Looks at 3 pixels from Layer 2 → which look at 5 pixels from Layer 1 → which look at **$7 \times 7$** from the image.
- $RF = 7$   *(Formula: $1 + 2(3) = 7$)*

> **The Formula:** For $3 \times 3$ kernels at stride 1:
> $$RF = 1 + 2 \times \text{depth}$$
> *(Base pixel = 1, plus 2 new pixels gained per layer $\times$ depth)*

### Feature Hierarchy
- **Early layers**: Edges, gradients, colors
- **Middle layers**: Textures, corners, circles
- **Deep layers**: Object parts (eyes, wheels, text)
- Network learns this hierarchy automatically through backpropagation

---

## 7. **Channel Operations**

### Multiple Output Channels
- Need multiple filters to detect different patterns simultaneously (e.g., vertical edges, horizontal edges, textures)
- **Filter bank**: `C_out` distinct kernels, each producing one feature map.

**Parameter Count Formula:**
```
Params = (K² × C_in + 1) × C_out
```

**Term-by-Term Breakdown:**

| Term | Meaning | Why? |
|---|---|---|
| `K²` | Spatial weights | A K×K filter has K² spatial pixels (e.g., 3×3 = 9 weights) |
| `× C_in` | Input channels | The filter extends through *all* input channels (e.g., RGB = 3) |
| `+ 1` | Bias term | Each filter gets exactly *one* bias scalar added to its output |
| `× C_out` | Number of filters | You repeat this entire structure for every new filter you want |

**Example:** A `3x3` Conv layer taking an RGB image (`C_in=3`) and producing `64` feature maps (`C_out=64`):
- One filter: `(3 × 3 × 3) + 1` = `28` parameters
- Whole layer: `28 × 64` = `1,792` parameters

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

**1. Parameter efficiency (Weight Sharing)**
- **MLP:** A 200x200 image has 40,000 pixels. Connecting them to just 1,000 hidden neurons requires **40 million** parameters ($40,000 \times 1,000$).
- **CNN:** A $3 \times 3$ filter applied across the whole image requires only **9** parameters. By sharing the same filter weights everywhere, CNNs drastically reduce memory footprint and overfitting.

**2. Translation equivariance/invariance**
- **MLP:** A cat in the top-left and a cat in the bottom-right are treated as completely different inputs.
- **CNN:** Because the same filter slides across the whole image, if it learns to detect a "cat ear" in the top-left, it automatically knows how to detect it in the bottom-right.

**3. Hierarchical learning**
- **MLP:** Tries to learn the complex mapping from raw pixels to "cat" all at once in fully-connected layers.
- **CNN:** Naturally builds a composition of features layer-by-layer.
  - Layer 1: Edges and colors
  - Layer 2: Textures and shapes (corners, circles)
  - Layer 3: Object parts (eyes, wheels)
  - Layer 4: Full objects (faces, cars)

**4. Spatial structure preservation**
- **MLP:** The very first step is to "flatten" a 2D image into a 1D vector (e.g., $28 \times 28 \rightarrow 784$). This completely destroys the geometric relationship between neighbor pixels.
- **CNN:** Keeps the data in 2D (or 3D with channels) throughout the network. It explicitly knows that pixel $(10,10)$ is right next to pixel $(10,11)$.

**5. Inductive bias (Locality & Stationarity)**
- **Inductive Bias** is the set of assumptions a model makes to learn faster. CNNs assume:
  - **Locality:** Pixels close to each other are strongly correlated (e.g., a pixel of a dog's fur is likely next to another pixel of fur). A pixel on the opposite side of the image probably has no direct relationship.
  - **Stationarity:** A pattern (like an edge) has the same meaning no matter where it appears in the image.

---

## Limitations

While CNNs are powerful, they have strict geometric blind spots:

**1. Not rotation invariant**
- A CNN that learns to recognize a vertical "3" will completely fail to recognize it if the image is rotated 90 degrees.
- **Fix:** Data augmentation (you must manually rotate training images so the network learns all orientations).

**2. Not scale invariant**
- If a CNN learns to detect a face that takes up the *whole* image, it will miss a tiny face in the background. The filters are fixed in size ($3 \times 3$, $5 \times 5$, etc.).
- **Fix:** You must train it on images with objects at multiple distances/scales.

**3. Limited geometric transformations**
- CNNs struggle with complex deformations (e.g., a crumpled piece of paper vs. a flat one). They expect pixels to roughly stay in their standard grid structure.

**4. Require large datasets**
- Because CNNs learn everything from scratch (edges $\rightarrow$ shapes $\rightarrow$ objects), they require tens of thousands of examples to build robust feature extractors.

**5. Black box (Interpretability)**
- While we can visualize early layers (which look like edge detectors), the deep layers (Layer 50) become chaotic mixtures of patterns that humans cannot understand. We know *that* it predicted "dog," but it's hard to prove *why*.

---

## Conclusion

CNNs revolutionized computer vision because they stopped treating images as random lists of numbers (like MLPs did) and started respecting the **physics of vision**. They encode four massive spatial assumptions (inductive biases):

1. **Local connectivity (Locality):** The world is made of local parts. An eye is a cluster of pixels; it's not scattered randomly across the image.
2. **Weight sharing (Translation equivariance):** The laws of physics don't change based on where you look. An edge in the top-left is identical to an edge in the bottom-right.
3. **Pooling (Translation invariance):** We care *that* an object is present, not its exact coordinate down to the millimeter.
4. **Hierarchical architecture (Feature composition):** The visual world is compositional. Pixels form edges $\rightarrow$ edges form shapes $\rightarrow$ shapes form objects.

By hardcoding these assumptions into the architecture, CNNs learn much faster, generalize exponentially better, and run with drastically fewer parameters. They are the foundation of modern computer vision.

---

## Next Steps
- **Study specific architectures:** See how these basic blocks were scaled up historically (LeNet, AlexNet, VGGNet, ResNet).
- **Understand advanced concepts:** Learn how skip connections (ResNets) let us build networks 100+ layers deep, and how attention mechanisms help the network focus on specific parts of the image.
- **Explore applications:** Move beyond simple classification ("is there a dog?") into **object detection** ("draw a box around the dog") and **segmentation** ("trace the exact outline of the dog").
