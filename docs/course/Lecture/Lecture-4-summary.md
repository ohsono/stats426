# Modern Architectures for CNNs

**Lecturer:** George Michailidis  
**Course:** STAT 426  
**Topic:** Modern Architectures for CNNs  

## Overview
This lecture explores the evolution of Convolutional Neural Network (CNN) architectures, highlighting key innovations that enabled deeper networks and improved performance on complex visual tasks.

---

## 1. LeNet (The Pioneer)
**Origin:** Proposed by Yann LeCun et al. (1990s) for handwritten digit recognition (MNIST).

*   **Motivation:** Overcome limitations of Multilayer Perceptrons (MLPs) such as loss of spatial structure and parameter explosion.
*   **Key Concepts:**
    *   **Local Interactions:** Convolutions focus on small local regions.
    *   **Translation Invariance:** Features are recognizable anywhere in the image.
    *   **Parameter Sharing:** Same weights used across the image.
*   **Architecture:**
    *   Alternating Convolutional (5x5) and Average Pooling (2x2) layers.
    *   Sigmoid activation.
    *   Ends with fully connected layers.

---

## 2. AlexNet (The Revolution)
**Origin:** Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton (2012). Won the ImageNet Challenge.

*   **Significance:** Proved deep CNNs could scale to realistic high-resolution photography.
*   **Key Innovations:**
    *   **ReLU Activation:** Solved vanishing gradients, faster training than Sigmoid/Tanh.
    *   **Dropout:** Reduced overfitting in fully connected layers.
    *   **Data Augmentation:** Flips, crops, jitter to expand dataset.
    *   **GPUs:** Enabled training of massive models.
*   **Architecture:**
    *   Deeper (8 layers) and wider than LeNet.
    *   Learned features from large kernels (11x11) down to fine textures (3x3).
    *   Used Max Pooling instead of Average Pooling.

---

## 3. VGG (Design Patterns)
**Origin:** Visual Geometry Group at Oxford (2014).

*   **Philosophy:** "Deep and Narrow".
*   **Key Innovation:**
    *   Replaced large kernels (11x11, 5x5) with stacks of small **3x3 kernels**.
    *   **Benefits:**
        *   Same receptive field with fewer parameters.
        *   More non-linearity (more ReLU layers) allows learning complex features.
*   **Architecture (VGG-11, VGG-16, etc.):**
    *   Repeating blocks of 3x3 Convs (pad 1, stride 1) followed by 2x2 Max Pooling.
    *   Spatial dim shrinks while depth increases.

---

## 4. Network-in-Network (NiN)
**Origin:** Lin et al. (2013).

*   **Motivation:** Standard linear filters are too simple for complex local patches.
*   **Key Innovation:**
    *   **Micro-Networks (1x1 Convs):** Embeds a small MLP (via 1x1 convolutions) at every pixel location.
    *   **1x1 Convolution:** Acts as a linear projection across channels (pixel-wise dense layer). Enables complex cross-channel feature mixing.
    *   **Global Average Pooling (GAP):** Replaces the heavy fully connected layers at the end. Averages each feature map to get a class score. Drastically reduces parameters and overfitting.

---

## 5. GoogLeNet (Inception)
**Origin:** Google (Szegedy et al., 2014).

*   **Motivation:** Go deeper without computational explosion.
*   **Key Innovation: Inception Block**
    *   Instead of choosing one filter size (1x1, 3x3, or 5x5), use **all of them in parallel** and concatenate results.
    *   Allows the network to capture multi-scale features.
*   **Efficiency:**
    *   **1x1 "Bottlenecks":** Used *before* expensive 3x3 or 5x5 convolutions to reduce channel dimensionality (e.g., squashing 192 channels to 16).
    *   Resulted in a 22-layer network with only 7 million parameters (vs. AlexNet's 60M).

---

## 6. Batch Normalization
**Origin:** Ioffe and Szegedy (2015).

*   **Problem:** "Internal Covariate Shift" - distribution of layer inputs changes during training, making optimization hard.
*   **Solution:** Normalize layer inputs for every mini-batch (mean 0, variance 1).
    *   Includes learnable scale ($\gamma$) and shift ($\beta$) parameters.
*   **Benefits:**
    *   Faster convergence (higher learning rates).
    *   Less sensitivity to initialization.
    *   Acts as a regularizer.

---

## 7. ResNet (Residual Learning)
**Origin:** He et al. (2015).

*   **Challenge:** Deeper networks suffered from degradation (higher error) due to vanishing gradients.
*   **Key Innovation: Residual Block**
    *   Learns the residual mapping $g(x) = f(x) - x$ instead of direct mapping.
    *   **Shortcut Connection:** Passes input $x$ directly to output ($f(x) + x$).
    *   Allows gradients to flow easily ("superhighways") through the network.
*   **Impact:** Enabled training of networks with 100+ (even 1000+) layers.

---

## 8. DenseNet (Feature Reuse)
**Origin:** Huang et al. (2017).

*   **Concept:**
    *   **Dense Block:** Each layer receives inputs from *all* previous layers (concatenation, not summation like ResNet).
    *   **Feature Reuse:** Extremely efficient; early features are directly accessible to later layers.
*   **Architecture:**
    *   **Growth Rate ($k$):** Number of new features added per layer.
    *   **Transition Layers:** Use 1x1 Conv and Pooling to compress channels and spatial size between dense blocks.
*   **Benefits:**
    *   Parameter efficiency (often better than ResNet).
    *   Strong gradient flow.

---

## Summary of Evolution
1.  **LeNet:** Convolution + Pooling + FC.
2.  **AlexNet:** Bigger, deeper, ReLU, Dropout, GPU.
3.  **VGG:** Modular, small 3x3 kernels only.
4.  **NiN:** 1x1 Convs (MLP per pixel) + Global Average Pooling.
5.  **GoogLeNet:** Multi-branch Inception blocks + 1x1 Bottlenecks.
6.  **ResNet:** Skip connections (Add) to solve vanishing gradients.
7.  **DenseNet:** Dense connections (Concat) for max feature reuse.
