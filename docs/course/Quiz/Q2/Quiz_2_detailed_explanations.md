# STAT 426: Quiz 2 - Complete Explanations with All Answer Options

**Based on:** Lectures 3 & 4

---

## Q1: Why does flattening fail?

**Correct Answer:** **(B) It discards geometric relationships**

### Why (B) is correct:
Flattening destroys the 2D grid structure of images. When you flatten a 2D image into a 1D vector, spatially adjacent pixels like (i, j) and (i+1, j) become distant in the vector. This breaks the local connectivity that's crucial for vision tasks. **Reference: Lec 3, Slide 4**

### Why the other answers are wrong:

**(A) It increases computational cost** - INCORRECT
- Flattening actually reduces computational cost in terms of simple operations. However, it requires vastly more parameters because you lose parameter sharing. The issue isn't computational cost per se—it's about losing spatial structure.

**(C) It changes color channels** - INCORRECT
- Flattening doesn't change the color values; it just reorganizes them into a 1D array. The actual color information is preserved.

**(D) It forces linear activations** - INCORRECT
- Flattening doesn't restrict the network to linear activations. You can still use ReLU, sigmoid, or any activation function after flattening. This is completely unrelated to why flattening fails for images.

---

## Q2: Permutation Failure Experiment

**Correct Answer:** **(A) Randomly permuting pixels makes the image unrecognizable to humans, but an MLP yields identical performance**

### Why (A) is correct:
This thought experiment is central to understanding why CNNs exist. If you randomly scramble all pixel positions:
- Humans see gibberish (no spatial structure)
- An MLP trained on the scrambled data will achieve the same accuracy as on the original data

This proves MLPs treat pixels as **independent features** with no awareness of spatial topology. **Reference: Lec 3, Slide 5**

### Why the other answers are wrong:

**(B) Weight permutation affects CNN more** - INCORRECT
- The permutation failure experiment is about permuting *pixels*, not weights. Additionally, scrambling pixel positions would devastate a CNN's performance (because it destroys spatial patterns), while an MLP remains unaffected. The claim is backwards.

**(C) Training image order causes failure** - INCORRECT
- The order of training images doesn't determine the core insight. Whether you shuffle training data or not, MLPs still can't capture spatial structure that's fundamental to image recognition.

**(D) Permuting channels prevents learning edges** - INCORRECT
- The experiment is about permuting pixel locations, not channels. Also, edge detection depends on spatial relationships, which is exactly why MLPs fail at this task.

---

## Q3: Weight Sharing Benefit

**Correct Answer:** **(C) It drastically reduces the number of parameters**

### Why (C) is correct:
Weight sharing means using a single **K×K kernel** that scans across the entire input, reusing the same weights. Instead of learning separate weights for every position (which would require millions of parameters), you learn one kernel and apply it everywhere.

Example: A 3×3 kernel with 9 parameters, applied across a 224×224 image, is vastly more efficient than a fully connected layer requiring 224² × 224² parameters.

**Reference: Lec 3, Slides 10 & 22**

### Why the other answers are wrong:

**(A) Ensures positive output** - INCORRECT
- Weight sharing has nothing to do with ensuring positive outputs. That's controlled by activation functions (like ReLU). Weight sharing is purely about reusing weights across space.

**(B) Allows processing 3D images** - INCORRECT
- Weight sharing doesn't enable 3D image processing. In fact, 3D convolutions (for video or volumetric data) still use weight sharing, but that's not the primary benefit cited here. The benefit is parameter efficiency.

**(D) Eliminates bias term** - INCORRECT
- Weight sharing and bias terms are independent concepts. You can have weight sharing with or without bias. Bias terms are necessary for learning proper thresholds, regardless of whether you share weights.

---

## Q4: Translation Equivariance vs Invariance

**Correct Answer:** **(D) Equivariance means if the input shifts, the feature map shifts; Invariance means output stays the same**

### Why (D) is correct:
This is a critical distinction in modern CNNs:

- **Translation Equivariance** (Convolution): If you shift the input by k pixels, the feature map also shifts by k pixels. Features move with the input.
- **Translation Invariance** (Pooling): Small spatial shifts in the input don't significantly change the pooled output. It provides robustness to position changes.

Modern CNNs use **both strategically**: convolutions build equivariant features, pooling adds invariance.

**Reference: Lec 3, Slide 23**

### Why the other answers are wrong:

**(A) Reversed definitions** - INCORRECT
- This answer has the definitions backwards. Equivariance is NOT about output staying constant. That's invariance.

**(B) Convolutional is Invariant** - INCORRECT
- Convolutional layers are **equivariant**, not invariant. Pooling layers provide invariance. This answer has them completely switched.

**(C) Both mean the same thing** - INCORRECT
- These are fundamentally different concepts. Equivariance and invariance are opposite properties. Equivariance tracks changes; invariance ignores them.

---

## Q5: Convolution vs Cross-Correlation

**Correct Answer:** **(C) Convolution requires the kernel to be flipped, while cross-correlation does not**

### Why (C) is correct:
Mathematically:
- **True Convolution**: Flips the kernel both horizontally and vertically before applying it
- **Cross-Correlation**: Applies the kernel directly without flipping

Deep learning libraries (TensorFlow, PyTorch) implement cross-correlation but call it "convolution" by convention. In practice, this doesn't matter much because during backpropagation, the network learns equivalent filters regardless of whether we flip or not.

**Reference: Lec 3, Slide 16**

### Why the other answers are wrong:

**(A) Cross-correlation requires flipping** - INCORRECT
- This is backwards. Cross-correlation does NOT flip; convolution does.

**(B) Cross-correlation is faster** - INCORRECT
- Both operations have similar computational complexity. The difference is mathematical, not computational speed. Both are O(H×W×K²×C) in time complexity.

**(D) Convolution only for grayscale** - INCORRECT
- Convolution works on images of any number of channels (grayscale, RGB, etc.). The mathematical operation is channel-agnostic.

---

## Q6: Output Size Calculation

**Correct Answer:** **(B) 28 × 28**

**Formula:** O = (W - K + 2P) / S + 1

**Calculation:** O = (32 - 5 + 2(0)) / 1 + 1 = 28

**Reference: Lec 3, Slide 42**

### Why the other answers are wrong:

**(A) 32 × 32** - INCORRECT
- This assumes no change in size (no convolution effect). With a 5×5 kernel and stride 1, you definitely lose spatial dimensions. 32 = 32 would only be true with appropriate padding to maintain size.

**(C) 27 × 27** - INCORRECT
- This would result from: (32 - 5) / 1 + 1 = 28. Wait, that's actually 28. The formula (32 - 5) = 27, then +1 = 28. This answer might result from forgetting the "+1" in the formula or miscalculation.

**(D) 16 × 16** - INCORRECT
- This would suggest dividing 32 by 2, implying a stride of 2 or aggressive downsampling. The problem specifies stride = 1, so this is wrong.

---

## Q7: "Same" Padding Purpose

**Correct Answer:** **(A) To ensure the output feature map has the same spatial dimensions as the input**

### Why (A) is correct:
"Same" padding adds border pixels (typically zeros) around the input so that after convolution with stride 1, the output has the same H × W as the input.

Without padding: Output shrinks
With "Same" padding: Output = Input size

This is crucial for building deep networks where you want to preserve spatial dimensions while stacking layers and increasing depth.

**Reference: Lec 3, Slides 34-39**

### Why the other answers are wrong:

**(B) Ensures same weights in every layer** - INCORRECT
- Padding has nothing to do with weight reuse across layers. Each layer learns its own weights independently.

**(C) Forces square output** - INCORRECT
- Padding can be applied to non-square inputs (e.g., 224×512) to produce non-square outputs of the same size. It doesn't force squares.

**(D) Same activation function** - INCORRECT
- Padding and activation functions are independent. You can use any activation function regardless of padding type.

---

## Q8: Bias Term Role

**Correct Answer:** **(B) It acts as a threshold, allowing a filter to activate even when input values are zero or weak**

### Why (B) is correct:
The bias term shifts the activation function. Without bias, every filter needs positive input to fire (activate). The bias allows each filter to set its own threshold for activation.

Example: z = w₁x₁ + w₂x₂ + ... + **b** → activation(z)

The bias **b** lets the neuron learn "I should activate when the weighted sum exceeds **-b**" rather than always requiring positive inputs.

**Reference: Lec 3, Slide 28**

### Why the other answers are wrong:

**(A) Shifts the kernel position** - INCORRECT
- Stride determines kernel position shifts, not bias. Bias shifts the threshold, not the spatial position.

**(C) Prevents vanishing gradients** - INCORRECT
- That's the role of activation functions like ReLU. Bias doesn't directly address vanishing gradients (though it can indirectly help by enabling better thresholds).

**(D) Determines stride** - INCORRECT
- Stride is a separate hyperparameter. Bias has nothing to do with determining stride values.

---

## Q9: Stride and Padding Calculation

**Correct Answer:** **(D) 14 × 20**

**Input:** 28 × 40, **Kernel:** 3×3, **Padding:** P=1, **Stride:** S=2

**Calculations:**
- Height: (28 - 3 + 2(1)) / 2 + 1 = 27/2 + 1 = 13.5 + 1 = 14.5 → 14
- Width: (40 - 3 + 2(1)) / 2 + 1 = 39/2 + 1 = 19.5 + 1 = 20.5 → 20

**Reference: Lec 3, Slides 31 & 42**

### Why the other answers are wrong:

**(A) 13 × 19** - INCORRECT
- This might result from incorrect rounding or not properly applying the formula. The correct calculation yields 14 × 20, not 13 × 19.

**(B) 28 × 40** - INCORRECT
- This assumes no change (stride 1, appropriate padding). With stride 2, you're downsampling by 2, so both dimensions should be approximately halved.

**(C) 15 × 21** - INCORRECT
- This might result from using S=2 but applying the formula incorrectly or rounding up instead of down. The correct answer is 14 × 20.

---

## Q10: Stride Effect

**Correct Answer:** **(C) It performs downsampling, skipping spatial locations to trade resolution for efficiency**

### Why (C) is correct:
Stride determines the step size when sliding the kernel across the input. Stride S=2 means you skip every other position, sampling the input at intervals of 2. This effectively reduces spatial resolution by a factor of S.

This is a key technique for:
- Reducing computation (fewer output positions to compute)
- Capturing higher-level features (each neuron sees a larger patch)
- Building hierarchical representations

**Reference: Lec 3, Slide 40**

### Why the other answers are wrong:

**(A) Increases spatial resolution** - INCORRECT
- Stride increases spatial resolution? Exactly opposite. Large strides → smaller outputs → lower resolution.

**(B) Adds padding** - INCORRECT
- Padding and stride are independent parameters. Stride determines the step size; padding adds borders. They're not related.

**(D) Increases channels** - INCORRECT
- Stride affects spatial dimensions (height, width), not the channel dimension. The number of channels is determined by the number of kernels in the layer.

---

## Q11: Max Pooling Function

**Correct Answer:** **(C) To summarize a region by keeping only the strongest signal/activation**

### Why (C) is correct:
Max pooling divides the feature map into non-overlapping (or overlapping) windows and keeps only the maximum value in each window. This:
- Captures the most prominent feature presence
- Reduces spatial dimensions
- Provides some translation invariance
- Reduces parameters in the network

Think of it as: "Is this important feature present in this region?" If yes (high activation), keep it. The specific location is less important than presence.

**Reference: Lec 3, Slide 49**

### Why the other answers are wrong:

**(A) Averages noise** - INCORRECT
- That's average pooling, not max pooling. Max pooling selects the maximum, which actually keeps strong signals and ignores weak noise. Average pooling would smooth everything.

**(B) Increases parameters** - INCORRECT
- Max pooling reduces parameters by downsampling. It doesn't add parameters; it removes spatial dimensions.

**(D) Reconstructs original image** - INCORRECT
- Max pooling is lossy and irreversible. Once you keep only the max values, you can't reconstruct the original feature map. That would require unpooling (used in autoencoders/deconvnets).

---

## Q12: Global Average Pooling (GAP)

**Correct Answer:** **(C) It removes positional information and drastically reduces the parameter count, preventing overfitting**

### Why (C) is correct:
GAP computes the average of all spatial locations for each channel, producing a vector of length = number of channels.

Benefits:
- **Removes spatial position info**: The network doesn't care WHERE a feature appears, only WHETHER it appears
- **Drastically reduces parameters**: Instead of flattening all spatial locations (which leads to huge dense layers), you get one value per channel
- **Prevents overfitting**: Fewer parameters = better generalization
- **Enables variable input sizes**: Unlike fully connected layers, GAP works on any input size

Introduced in Network-in-Network (NiN) architecture.

**Reference: Lec 3, Slide 54**

### Why the other answers are wrong:

**(A) Retains position more accurately** - INCORRECT
- GAP does the opposite: it *discards* position information. This is a feature, not a bug.

**(B) Increases dimensionality** - INCORRECT
- GAP drastically *reduces* dimensionality. A 512-channel feature map becomes a 512-d vector, compared to flattening which would be much larger.

**(D) Faster than dot product** - INCORRECT
- GAP and fully connected layers don't have a direct computational speed comparison that makes this relevant. The reason GAP is preferred is architectural (parameter reduction), not speed.

---

## Q13: Receptive Field Expansion

**Correct Answer:** **(C) It expands, allowing deeper units to indirectly "see" a larger portion of the original input**

### Why (C) is correct:
The **receptive field** is the size of the input region that influences a particular neuron's output.

- Layer 1 conv: Each neuron sees only a K×K patch
- Layer 2 conv: Each neuron sees a larger patch (its kernel sees multiple layer-1 neurons, each seeing a patch)
- Deeper layers: Cumulative effect means neurons can see larger regions of the original input

Example with 3×3 kernels and stride 1:
- Layer 1: RF = 3×3
- Layer 2: RF = 5×5
- Layer 3: RF = 7×7

This is why we can use small kernels—the receptive field grows naturally.

**Reference: Lec 3, Slide 58**

### Why the other answers are wrong:

**(A) Stays same as kernel** - INCORRECT
- If receptive field stayed 3×3 forever, networks couldn't capture global context. The entire point of depth is to expand the receptive field.

**(B) Shrinks for finer details** - INCORRECT
- Actually backwards. Deeper layers see LARGER patches (lower resolution features), while shallow layers see small patches (fine details). Depth increases RF.

**(D) Becomes random** - INCORRECT
- Receptive field expansion is deterministic and predictable. It follows a mathematical formula based on kernel size, stride, and layer depth.

---

## Q14: Biological Inspiration (Hubel & Wiesel)

**Correct Answer:** **(A) "Simple Cells" and "Complex Cells" in the visual cortex**

### Why (A) is correct:
Hubel & Wiesel's 1959 discovery of orientation-selective cells in cat visual cortex directly inspired CNN architecture:

- **Simple Cells**: Respond to oriented edges in specific locations. These inspired **convolutional layers** that detect local patterns.
- **Complex Cells**: Respond to oriented edges but are robust to small shifts in position. These inspired **pooling layers** that provide translation invariance.

This biological hierarchy (simple → complex → progressively invariant) became the foundation for building CNNs as stacks of conv and pooling layers.

**Reference: Lec 3, Slide 59**

### Why the other answers are wrong:

**(B) RGB processing in retina** - INCORRECT
- While the retina does process color, this discovery isn't what inspired CNNs. The key insight was about edge detection and invariance, not color processing.

**(C) Brain uses backpropagation** - INCORRECT
- Backpropagation is a mathematical learning algorithm. Hubel & Wiesel discovered how the brain *recognizes* patterns, not how it learns them via backpropagation (which isn't biologically plausible anyway).

**(D) Neurons fire randomly during sleep** - INCORRECT
- This is completely unrelated to CNN architecture. Hubel & Wiesel's discovery was about awake visual processing, not sleep.

---

## Q15: 1×1 Convolution Purpose

**Correct Answer:** **(B) To perform dimensionality reduction or expansion along channel depth without changing spatial resolution (H × W)**

### Why (B) is correct:
A 1×1 convolution is a special case: it operates on the channel dimension, not spatial dimensions.

With K filters of 1×1:
- Input: H × W × C (height × width × channels)
- Output: H × W × K (height and width unchanged, channels = K)

Uses:
- **Dimensionality reduction**: 192 → 16 channels (bottleneck in Inception)
- **Channel mixing**: Learn non-linear combinations across channels
- **Equivalent to pixel-wise dense layer**: Applies a dense transformation to the channel vector at every pixel

**Reference: Lec 3, Slides 67 & 68 (and crucial for Lec 4, GoogLeNet)**

### Why the other answers are wrong:

**(A) Shifts image by 1 pixel** - INCORRECT
- 1×1 convolutions don't shift or translate the image. They're purely channel operations on each spatial location independently.

**(C) Pass-through layer** - INCORRECT
- 1×1 convolutions are highly non-trivial. They learn to mix channels, apply non-linearity, and often compress high-dimensional features.

**(D) Removes noise** - INCORRECT
- That's not the purpose of 1×1 convolutions. They're for channel-wise learning, not noise removal (which would be filtering/smoothing).

---

## Q16: Filter Bank Definition

**Correct Answer:** **(B) The collection of multiple kernels learned in a single layer, where each kernel detects a distinct feature**

### Why (B) is correct:
A "filter bank" or "kernel bank" is the set of all K kernels in a single convolutional layer.

- Each kernel learns different features (edges at different orientations, textures, patterns)
- If a layer has 64 filters, it learns 64 different feature detectors
- Output has 64 channels (one per filter)

The "bank" metaphor: Just as a bank stores money, a layer stores a bank of filters, each specializing in different features.

**Reference: Lec 3, Slides 64 & 65**

### Why the other answers are wrong:

**(A) Repository of pre-trained weights** - INCORRECT
- A filter bank is not a repository or storage system. It's the set of actively used kernels in the current layer.

**(C) Hardware acceleration unit** - INCORRECT
- Filter bank is a software/algorithm concept, not a hardware component. Though GPUs accelerate filter banks, a filter bank itself isn't hardware.

**(D) Set of validation images** - INCORRECT
- Completely unrelated. Validation images are data; filter banks are learned weights. No connection.

---

## Q17: End-to-End Learning in Deep CNNs

**Correct Answer:** **(A) Traditional methods required manual feature engineering (e.g., Sobel filters), whereas CNNs learn optimal filters directly from data via backpropagation**

### Why (A) is correct:
Before deep learning (pre-2012):
- Manually designed filters: Sobel (edges), SIFT, HOG features
- Hand-crafted feature extraction
- Separate classifier (SVM, etc.)

With deep CNNs (post-AlexNet 2012):
- Raw pixels → CNN learns all features automatically
- Features emerge from backpropagation without manual engineering
- End-to-end: pixels to class predictions, all learned

LeNet demonstrated this possibility; AlexNet proved it works at scale.

**Reference: Lec 3, Slide 27**

### Why the other answers are wrong:

**(B) Test set training first** - INCORRECT
- That's bad practice (data leakage), not "end-to-end learning." End-to-end has nothing to do with training procedure order.

**(C) Traditional methods used neural nets** - INCORRECT
- Traditional methods (pre-2012 deep learning) mostly used hand-crafted features with shallow classifiers (SVM). Neural networks existed but weren't dominant.

**(D) Requires flattening input** - INCORRECT
- End-to-end learning has no requirement to flatten. In fact, CNNs preserve spatial structure specifically to avoid flattening.

---

## Q18: Modern CNN Architecture Evolution

**Correct Answer:** **(C) Spatial dimensions (H, W) decrease (downsampling), while depth (number of channels) increases**

### Why (C) is correct:
Standard CNN progression (VGG, ResNet, etc.):
- **Early layers**: Large spatial dimensions, few channels (e.g., 224×224×64)
- **Middle layers**: Medium spatial dimensions, more channels (e.g., 56×56×256)
- **Deep layers**: Small spatial dimensions, many channels (e.g., 7×7×512)

Strategy:
- **Spatial reduction** (via pooling/stride): Reduces computation, increases receptive field
- **Channel expansion**: More diverse feature representations
- **Progressive abstraction**: Low-level details → high-level concepts

This is why VGG uses repeated blocks: Conv(3×3) → Conv(3×3) → Pool(2×2), progressively building this pattern.

**Reference: Lec 4, Slides 15-18**

### Why the other answers are wrong:

**(A) H, W increase; channels decrease** - INCORRECT
- This would make the network progressively lose information and spatial detail. It's the opposite of what happens.

**(B) H, W and channels stay same** - INCORRECT
- If dimensions never changed, you couldn't build hierarchical representations or reduce computation. Modern CNNs rely on progressive downsampling.

**(D) Becomes 1D immediately** - INCORRECT
- Flattening happens at the very end, after all convolutional layers. The network maintains spatial structure through most of its depth.

---

## Q19: AlexNet's Innovations

**Correct Answer:** **(C) The use of Residual Skip Connections**

### Why (C) is correct:
AlexNet (2012) introduced:
- ✓ **ReLU activation** instead of Sigmoid/Tanh
- ✓ **Dropout regularization** to prevent overfitting
- ✓ **GPU training** to handle massive datasets

What AlexNet did NOT introduce:
- ✗ **Residual skip connections** — These were introduced **7 years later by ResNet (2015)**

ResNet's key innovation was showing that skip connections allow training extremely deep networks (>100 layers) by providing gradient superhighways.

**Reference: Lec 3, Slide 10 (AlexNet); Lec 4, Slide 37 (ResNet)**

### Why the other answers are wrong:

**(A) ReLU activation** - INCORRECT
- AlexNet DID introduce ReLU. This is a correct statement, but the question asks what was NOT introduced by AlexNet. We want the false statement.

**(B) Dropout regularization** - INCORRECT
- AlexNet DID use Dropout heavily in fully connected layers. This is another correct innovation, not the false one.

**(D) GPU training** - INCORRECT
- AlexNet's ability to train on ImageNet with GPUs was a major contribution. The paper's practical impact came partly from this hardware advance.

---

## Q20: Batch Normalization Purpose

**Correct Answer:** **(B) To fix "Internal Covariate Shift" by standardizing layer inputs, which stabilizes training and allows higher learning rates**

### Why (B) is correct:
**Internal Covariate Shift**: As weights update during training, the distribution of inputs to each layer shifts continuously, making optimization difficult.

**Batch Normalization solution**:
- Normalize each mini-batch to have mean 0, variance 1: (x - μ_batch) / σ_batch
- Learn scale and shift parameters (γ, β) to allow the network to set its own optimal distribution

**Benefits**:
- Faster convergence
- Enables higher learning rates
- Less sensitivity to weight initialization
- Acts as a regularizer

**Reference: Lec 4, Slides 47 & 48**

### Why the other answers are wrong:

**(A) Normalizes input images** - INCORRECT
- Batch Norm happens layer-by-layer, not just at the input. This describes pre-processing, not Batch Norm's actual function.

**(C) Compresses model size** - INCORRECT
- Batch Norm doesn't compress models for mobile deployment. That's quantization or pruning. Batch Norm adds learnable parameters (γ, β).

**(D) Converts to grayscale** - INCORRECT
- Completely unrelated. Batch Norm normalizes distributions, not color spaces.

---

## Q21: 1×1 Convolution Equivalence

**Correct Answer:** **(B) A Fully Connected (Dense) layer applied independently to the channel vector at every pixel location**

### Why (B) is correct:
Mathematically, a 1×1 convolution at position (i, j) is equivalent to:
- Taking the channel vector C at position (i, j)
- Applying a matrix multiplication (dense layer) to transform it to K dimensions
- This happens independently at every spatial location

So:
- Input: H × W × C
- Apply 1×1 convolution with K filters
- Output: H × W × K (where each pixel's K-d output is a dense transformation of its input C-d vector)

This insight (from Network-in-Network paper) showed that 1×1 convolutions can provide local non-linear feature mixing—like a micro MLP at every pixel.

**Reference: Lec 4, Slides 41 & 42**

### Why the other answers are wrong:

**(A) Gaussian Blur** - INCORRECT
- 1×1 convolutions have no spatial extent. Blurring requires spatial kernels (3×3 or larger). No blur possible with 1×1.

**(C) Max Pooling** - INCORRECT
- Max pooling selects the maximum value. 1×1 convolutions learn weighted combinations. Completely different operations.

**(D) Image rotation** - INCORRECT
- Geometric transformations like rotation require spatial context. 1×1 operations are purely channel-wise, not geometric.

---

## Q22: Inception Block Bottleneck

**Correct Answer:** **(B) To act as a "bottleneck" layer that reduces the channel dimension, drastically lowering the computational cost of subsequent filters**

### Why (B) is correct:
GoogLeNet's Inception module challenge:
- Naive approach: Use 1×1, 3×3, and 5×5 convolutions in parallel
- Problem: 5×5 convolutions are expensive. High-resolution feature maps make it worse.

**Bottleneck solution**:
- Before expensive 3×3 or 5×5 convolutions, add a 1×1 convolution that compresses channels
- Example: 192 channels → 16 channels (1×1) → then apply 5×5
- The 5×5 now operates on 16×W×H instead of 192×W×H
- **Computational cost reduced by 12× with minimal accuracy loss**

Result: 22-layer network with only 7M parameters (vs AlexNet's 60M).

**Reference: Lec 4, Slides 41 & 42**

### Why the other answers are wrong:

**(A) Increase channels for detail** - INCORRECT
- Bottlenecks do the opposite: they *reduce* channels. This is the entire point—compression, not expansion.

**(C) Smooth the image** - INCORRECT
- Bottlenecks don't smooth. They compress channel information. Smoothing requires spatial filtering (Gaussian blur).

**(D) Change resolution** - INCORRECT
- 1×1 convolutions don't change spatial resolution (H × W stays same). Bottlenecks only modify channel dimension.

---

## Summary of Key Concepts

### Spatial Operations
- **Convolution**: Detects local patterns (equivariant)
- **Pooling**: Reduces spatial dimensions (invariant)
- **Stride**: Controls downsampling
- **Padding**: Controls output size

### Channel Operations
- **1×1 Convolution**: Dense layer per pixel (dimensionality reduction)
- **Filter Bank**: Collection of learned kernels per layer

### Architectural Principles
- **Weight Sharing**: Reuses kernels across space (parameter efficiency)
- **Progressive Downsampling**: H×W decreases, channels increase
- **Receptive Field Expansion**: Deeper layers see larger input patches
- **End-to-End Learning**: Features learned from data, not hand-crafted

### Modern Techniques
- **Batch Normalization**: Stabilizes training by fixing internal covariate shift
- **Residual Connections**: Enable very deep networks via gradient superhighways
- **Bottlenecks**: Reduce computation via channel compression
- **Global Average Pooling**: Removes position info, reduces parameters

### Historical Evolution
1. **LeNet** → Convolution + Pooling foundation
2. **AlexNet** → ReLU, Dropout, GPU scaling
3. **VGG** → Small 3×3 kernels consistently
4. **NiN** → 1×1 convs + Global Average Pooling
5. **GoogLeNet** → Inception blocks + bottlenecks
6. **ResNet** → Skip connections (enables depth)
7. **DenseNet** → Dense connections (feature reuse)
