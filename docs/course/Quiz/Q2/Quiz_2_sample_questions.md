# STAT 426: Quiz 2 Answer Key

**Based on:** Lectures 3 & 4

---

## Questions and Answers

### Q1
**Correct Answer:** (b) It discards geometric relationships...

**Explanation:** Flattening destroys the 2D grid structure, meaning vertical neighbors `(i, j)` and `(i+1, j)` become far apart in the vector.  
*Reference: Lec 3, Slide 4*

---

### Q2
**Correct Answer:** (a) Randomly permuting pixels... MLP yields identical performance...

**Explanation:** The "Permutation Failure" experiment shows MLPs treat pixels as independent features, ignoring spatial topology, unlike CNNs or human vision.  
*Reference: Lec 3, Slide 5*

---

### Q3
**Correct Answer:** (c) Drastically reduces the number of parameters...

**Explanation:** Weight sharing allows the network to use a single kernel scanned across the image rather than separate weights for every pixel.  
*Reference: Lec 3, Slide 10 & 22*

---

### Q4
**Correct Answer:** (d) Equivariance means if the input shifts, the feature map shifts...

**Explanation:** Convolution is translation equivariant (features move with input), while Pooling provides translation invariance (output remains stable).  
*Reference: Lec 3, Slide 23*

---

### Q5
**Correct Answer:** (c) Convolution requires the kernel to be flipped...

**Explanation:** Deep learning libraries technically implement cross-correlation (no flip), but it is conventionally called convolution.  
*Reference: Lec 3, Slide 16*

---

### Q6
**Correct Answer:** (b) 28×28

**Explanation:** Formula: `O = (W - K + 2P) / S + 1` ⇒ `(32 - 5 + 0) / 1 + 1 = 28`.  
*Reference: Lec 3, Slide 42*

---

### Q7
**Correct Answer:** (a) To ensure the output feature map has the same spatial dimensions...

**Explanation:** "Same" padding adds border pixels so that Input Height/Width equals Output Height/Width.  
*Reference: Lec 3, Slide 34-39*

---

### Q8
**Correct Answer:** (b) It acts as a threshold...

**Explanation:** The bias term shifts the activation function, allowing the filter to fire (or not) relative to a learned threshold.  
*Reference: Lec 3, Slide 28*

---

### Q9
**Correct Answer:** (d) 14×20

**Explanation:** Height: `(28 - 3 + 2) / 2 + 1 = 14`. Width: `(40 - 3 + 2) / 2 + 1 = 20`.  
*Reference: Lec 3, Slide 31 & 42*

---

### Q10
**Correct Answer:** (c) It performs downsampling...

**Explanation:** Strides `S > 1` skip pixels, effectively reducing the spatial resolution of the feature map.  
*Reference: Lec 3, Slide 40*

---

### Q11
**Correct Answer:** (c) To summarize a region by keeping only the strongest signal...

**Explanation:** Max pooling selects the maximum value in the window, capturing the most prominent feature presence.  
*Reference: Lec 3, Slide 49*

---

### Q12
**Correct Answer:** (c) It removes positional information...

**Explanation:** GAP (Global Average Pooling) averages the entire feature map, reducing parameters and enforcing invariance to the feature's specific location.  
*Reference: Lec 3, Slide 54*

---

### Q13
**Correct Answer:** (c) It expands...

**Explanation:** As you go deeper, units "see" a larger patch of the original input due to the cumulative effect of convolutions.  
*Reference: Lec 3, Slide 58*

---

### Q14
**Correct Answer:** (a) Simple Cells and Complex Cells...

**Explanation:** Hubel & Wiesel's discovery of orientation-selective cells in the visual cortex inspired the hierarchy of simple (conv) and complex (pool) layers.  
*Reference: Lec 3, Slide 59*

---

### Q15
**Correct Answer:** (b) To perform dimensionality reduction...

**Explanation:** 1×1 convolutions change the channel depth (e.g., 192 → 16) without altering spatial dimensions.  
*Reference: Lec 3, Slides 67 & 68*

---

### Q16
**Correct Answer:** (b) The collection of multiple kernels...

**Explanation:** A layer learns a "bank" of K filters, resulting in an output volume with depth K.  
*Reference: Lec 3, Slides 64 & 65*

---

### Q17
**Correct Answer:** (a) ...CNNs learn the optimal filters directly from data...

**Explanation:** LeNet demonstrated that features can be learned from raw pixels, replacing manual engineering (e.g., Sobel).  
*Reference: Lec 3, Slide 27*

---

### Q18
**Correct Answer:** (c) The spatial dimensions decrease... depth increases.

**Explanation:** Standard architectures (like VGG) progressively downsample space (via pooling) while increasing channels.  
*Reference: Lec 4, Slide 15-18*

---

### Q19
**Correct Answer:** (c) The use of Residual Skip Connections.

**Explanation:** AlexNet introduced ReLU, Dropout, and GPU training. Residual connections were introduced later by ResNet.  
*Reference: Lec 3, Slide 10 (AlexNet); Lec 4, Slide 37 (ResNet)*

---

### Q20
**Correct Answer:** (b) To fix "Internal Covariate Shift"...

**Explanation:** Batch Norm stabilizes distributions of layer inputs, enabling faster training and higher learning rates.  
*Reference: Lec 4, Slides 47 & 48*

---

### Q21
**Correct Answer:** (b) A Fully Connected (Dense) layer...

**Explanation:** A 1×1 convolution is mathematically identical to applying a dense layer to the channel vector at each pixel.  
*Reference: Lec 4, Slides 41 & 42*

---

### Q22
**Correct Answer:** (b) To act as a "bottleneck" layer...

**Explanation:** Inception modules use 1×1 filters to compress channels before expensive 3×3 or 5×5 convolutions.  
*Reference: Lec 4, Slide 41 & 42*
