Modern Architectures for CNNs
Modern Architectures of CNNs
George Michailidis
gmichail@ucla.edu
STAT 426
George Michailidis Modern Architectures for CNNs 1 / 63
Modern Architectures for CNNs
Many of the building blocks presented in Section 3 emerged in the CNN
architectures discussed next.
George Michailidis Modern Architectures for CNNs 2 / 63
Modern Architectures for CNNs
LeNet
Introduction to LeNet
Origins: Proposed by Yann LeCun et al. in the 1990s (AT&T Bell Labs).
Purpose: Designed to recognize handwritten digits (MNIST dataset) for
reading zip codes on mail and numbers on checks.
Significance:
▶ It was the first successful deployment of Convolutional Neural Networks
(CNNs).
▶ It demonstrated that neural networks could learn features directly from raw
pixels, replacing hand-engineered feature extraction.
The specific architecture discussed is formally known as LeNet-5.
George Michailidis Modern Architectures for CNNs 3 / 63
Modern Architectures for CNNs
LeNet
Motivation: Why CNNs over MLPs?
The Problem with Multilayer Perceptrons (MLPs):
MLPs require flattening images into 1D vectors (28×28→784)
The strategy in Homework 1.
Loss of Spatial Structure: Adjacent pixels lose their spatial relationship.
Parameter Explosion: Fully connecting every pixel to every hidden neuron is
computationally expensive (does not scale to images with more complex
patterns).
The LeNet Solution:
Local Interactions: Convolutional layers focus on small regions (kernels).
Translation Invariance: A feature learned in one corner is recognizable
anywhere.
Parameter Sharing: The same kernel weights are used across the entire
image.
George Michailidis Modern Architectures for CNNs 4 / 63
Modern Architectures for CNNs
LeNet
LeNet Architecture Overview
LeNet consists of two distinct structural blocks:
1. Convolutional Encoder:
▶ Extracts spatial features.
▶ Consists of two alternating blocks of Convolutional layers and Pooling layers.
2. Dense Block (Classifier):
▶ Compiles features to make a prediction.
▶ Consists of three Fully Connected layers.
Data Flow:
Input→Conv→Pool→Conv→Pool→Flatten→FC→FC→Output
George Michailidis Modern Architectures for CNNs 5 / 63
Modern Architectures for CNNs
LeNet
The Convolutional Block Details
Input: 28×28 single-channel (greyscale).
Layer 1 (C1 & S2):
▶ Conv: 6 output channels, 5×5 kernel, padding=2. (Output: 28×28)
▶ Activation: Sigmoid (Historically used; modern nets use ReLU).
▶ Pooling: Average Pooling, 2×2, stride 2. (Output: 14×14)
Layer 2 (C3 & S4):
▶ Conv: 16 output channels, 5×5 kernel, no padding. (Output: 10×10)
▶ Pooling: Average Pooling, 2×2, stride 2. (Output: 5×5)
George Michailidis Modern Architectures for CNNs 6 / 63
Modern Architectures for CNNs
LeNet
The Dense Block (Classifier)
Flattening:
▶ The output of the last pooling layer (16 channels×5×5) is flattened into a
1D vector of size 400.
Fully Connected Layers:
1. FC 1: Input 400→Output 120 (Sigmoid).
2. FC 2: Input 120→Output 84 (Sigmoid).
3. Output Layer: Input 84→Output 10 (Softmax/Gaussian).
Prediction:
▶ 10 probabilities corresponding to digits 0 through 9.
George Michailidis Modern Architectures for CNNs 7 / 63
Modern Architectures for CNNs
LeNet
Dimensionality Tracking
Assuming a batch size of 1 and Input (1×28×28):
Layer Type Kernel/Stride Input Shape Output Shape
Conv2d 5×5, Pad 2 1×28×28 6×28×28
AvgPool2d 2×2, Stride 2 6×28×28 6×14×14
Conv2d 5×5, Pad 0 6×14×14 16×10×10
AvgPool2d 2×2, Stride 2 16×10×10 16×5×5
Flatten - 16×5×5 400
Linear - 400 120
Linear - 120 84
Linear - 84 10
George Michailidis Modern Architectures for CNNs 8 / 63
Modern Architectures for CNNs
AlexNet
The Deep Learning Revolution: AlexNet
Origin: Introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in
2012.
The Challenge: The ImageNet Large Scale Visual Recognition Challenge.
Significance:
▶ While LeNet (1995) worked on small digits (MNIST), it failed on realistic,
high-resolution photography.
▶ AlexNet achieved a massive reduction in error rate (almost half of the
runner-up), proving that deep CNNs could scale.
Key Enablers:
1. Data: ImageNet provided millions of labeled images.
2. Hardware: The use of GPUs (Graphics Processing Units) made training
massive models feasible.
George Michailidis Modern Architectures for CNNs 9 / 63
Modern Architectures for CNNs
AlexNet
Key Innovations (vs. LeNet)
AlexNet is “evolutionary” (same principles). but introduced critical changes for
scalability:
1. ReLU Activation (Rectified Linear Unit):
▶ Replaced the Sigmoid/Tanh functions used in LeNet.
▶ f(x) = max(0,x).
▶ Benefit: Prevents the vanishing gradient problem and accelerates training.
2. Dropout Regularization:
▶ Applied in the fully connected layers.
▶ Randomly zeroes out neurons during training.
▶ Benefit: Drastically reduces overfitting (crucial for 50M+ parameters).
3. Data Augmentation:
▶ Artificially expanded the dataset via random flips, crops, and color jitter.
George Michailidis Modern Architectures for CNNs 10 / 63
Modern Architectures for CNNs
AlexNet
Architecture Overview
AlexNet is significantly deeper and wider than LeNet. It consists of 8 layers:
5 Convolutional Layers:
▶ Layers 1, 2, and 5 are followed by Max Pooling (LeNet used Average Pooling).
▶ Kernels: Starts large (11×11) to capture global shapes, shrinking to 3×3 for
fine textures.
3 Fully Connected Layers:
▶ Two massive hidden layers with 4096 neurons each.
▶ Final output layer matches the 1000 classes of ImageNet.
George Michailidis Modern Architectures for CNNs 11 / 63
Modern Architectures for CNNs
AlexNet
The Convolutional Layers (Detail)
Layer 1:
▶ Input: 224×224×3 (RGB Images).
▶ Conv: 11×11 kernel, stride 4, 96 output channels.
▶ Note:Large stride drastically reduces spatial dimension early on.
Layer 2:
▶ Conv: 5×5 kernel, padding 2, 256 output channels.
▶ Followed by Max Pooling (3×3, stride 2).
Layers 3, 4, 5:
▶ Stacked directly without pooling in between (to build complex features).
▶ L3 & L4: 384 output channels.
▶ L5: 256 channels, followed by final Max Pool.
George Michailidis Modern Architectures for CNNs 12 / 63
Modern Architectures for CNNs
AlexNet
Dimensionality Tracking
Assuming Input Image of size 224×224:
Layer Kernel / Stride / Pad Output Channels Output Shape
Input - 3 (RGB) 224×224
Conv 1 11×11, S:4, P:1 96 54×54
MaxPool 1 3×3, S:2 96 26×26
Conv 2 5×5, S:1, P:2 256 26×26
MaxPool 2 3×3, S:2 256 12×12
Conv 3 3×3, S:1, P:1 384 12×12
Conv 4 3×3, S:1, P:1 384 12×12
Conv 5 3×3, S:1, P:1 256 12×12
MaxPool 3 3×3, S:2 256 5×5
Flatten - - 6,400
FC 1 Dropout (0.5) - 4,096
FC 2 Dropout (0.5) - 4,096
Output Softmax - 1,000
George Michailidis Modern Architectures for CNNs 13 / 63
Modern Architectures for CNNs
AlexNet
Summary: From LeNet to AlexNet
Feature LeNet (1995) AlexNet (2012)
Input Size 28×28 (Greyscale) 224×224 (Color)
Depth 5 Layers 8 Layers
First Kernel 5×5 11×11
Activation Sigmoid ReLU
Pooling Average Pooling Max Pooling
Regularization Weight Decay Dropout
Parameters ≈60 Thousand ≈60 Million
Takeaway: AlexNet took the architectural ideas of LeNet and scaled them up
using new hardware (GPUs) and better non-linearities (ReLU) to solve complex
real-world vision tasks.
George Michailidis Modern Architectures for CNNs 14 / 63
Modern Architectures for CNNs
AlexNet
From Ad-Hoc to Design Patterns: Visual Geometry Group
Origin: Proposed by the Visual Geometry Group (VGG) at Oxford (Simonyan
& Zisserman, 2014).
The Shift:
▶ AlexNet showed that deep networks work, but its architecture was somewhat
ad-hoc (irregular kernel sizes 11×11,5×5).
▶ VGG introduced the concept of Blocks: repeating patterns of layers.
Key Insight:
▶ Replace large kernels with stacks of small 3×3 kernels.
▶ This created the philosophy of “Deep and Narrow”.
George Michailidis Modern Architectures for CNNs 15 / 63
Modern Architectures for CNNs
AlexNet
Why only 3×3 Convolutions?
VGG replaced the large 5×5 and 7×7 kernels with stacks of 3×3 kernels. Why?
1. Same Receptive Field:
▶ Two stacked 3×3 layers see the same amount of pixels as one 5×5 layer.
▶ Three stacked 3×3 layers see the same as one 7×7 layer.
2. Fewer Parameters:
▶ 5×5 kernel params:25
▶ Two 3×3 kernels params: 3 2 + 32 = 9 + 9 =18
▶ Result: We can make the network deeper for the same cost.
3. More Non-Linearity:
▶ Two layers mean two ReLU functions instead of one, allowing the network to
learn more complex features.
George Michailidis Modern Architectures for CNNs 16 / 63
Modern Architectures for CNNs
AlexNet
3. The Fundamental Building Block
The architecture is built entirely by repeating this specific VGG Block:
VGG Block Structure
1. Convolution Layers:
▶ Kernel Size: 3×3
▶ Padding: 1 (Maintains height/width)
▶ Stride: 1
▶ Activation: ReLU
2. Max Pooling Layer:
▶ Kernel Size: 2×2
▶ Stride: 2 (Halves height/width)
Design Pattern: The spatial dimension shrinks (224→112→56. . .) while the
depth (output channels) increases (64→128→256. . .).
George Michailidis Modern Architectures for CNNs 17 / 63
Modern Architectures for CNNs
AlexNet
VGG-11 Architecture
The “VGG-11” (8 Conv + 3 FC layers) is composed of 5 Blocks:
1. Block 1: 1 Conv layer, 64 Channels.
2. Block 2: 1 Conv layer, 128 Channels.
3. Block 3: 2 Conv layers, 256 Channels.
4. Block 4: 2 Conv layers, 512 Channels.
5. Block 5: 2 Conv layers, 512 Channels.
After the blocks, the features are flattened and passed through:
Flatten→FC(4096)→FC(4096)→Output(1000)
George Michailidis Modern Architectures for CNNs 18 / 63
Modern Architectures for CNNs
AlexNet
5. Dimensionality Tracking (VGG-11)
Assuming Input Image of size 224×224:
Block Convolutions Channels Output Shape
Input - 3 224×224
Block 1 1 layer 64 112×112 (after pool)
Block 2 1 layer 128 56×56 (after pool)
Block 3 2 layers 256 28×28 (after pool)
Block 4 2 layers 512 14×14 (after pool)
Block 5 2 layers 512 7×7 (after pool)
Flatten - - 25,088 (512×7×7)
FC 1 Dropout (0.5) - 4,096
FC 2 Dropout (0.5) - 4,096
Output Softmax - 1,000
George Michailidis Modern Architectures for CNNs 19 / 63
Modern Architectures for CNNs
AlexNet
Summary & Trade-offs
Strengths:
Simplicity: Very easy to implement using loops (repeated blocks).
Generalization: VGG features transfer very well to other tasks (object
detection, segmentation).
Weaknesses:
Computationally Expensive: VGG-19 is extremely slow compared to modern
networks like ResNet.
Memory Heavy:
▶ The first FC layer connects 25,088 inputs to 4,096 outputs.
▶ That alone is≈102Million parameters.
▶ VGG-16 total size is over 500MB.
George Michailidis Modern Architectures for CNNs 20 / 63
Modern Architectures for CNNs
Network-in-Network
Network in Network (NiN) — Overview
Limit of Classic CNNs: They use linear filters followed by a simple
nonlinearity. This might be too simple to extract complex features from a
local patch.
The NiN Idea: Embed small neural networks (micro MLPs) inside
convolutional layers to make local feature extraction more expressive.
Implementation: These micro-networks are implemented via 1×1
convolutions.
Structural Shift: Replaces the “memory-hungry” fully connected layers at the
end with Global Average Pooling (GAP).
George Michailidis Modern Architectures for CNNs 21 / 63
Modern Architectures for CNNs
Network-in-Network
Motivation: Why 1×1 Convolutions?
Classic Conv: One kernel produces one feature at a time
(Input·Kernel+Bias). It is a linear local operator.
NiN Block: Treats each spatial pixel across all channels as a vector input to a
shared MLP.
The Equivalence:
▶ A 1×1 convolution withC out filters is mathematically identical to aFully
Connected (Dense) layerapplied to each pixel independently.
▶ Parameters:C in ×C out.
Benefit: It allows complex, nonlinear mixing of channel information without
changing the spatial dimensions (H×W).
George Michailidis Modern Architectures for CNNs 22 / 63
Modern Architectures for CNNs
Network-in-Network
The NiN Block: A ”Mini-Network” for Every Pixel
A NiN block replaces the standard Convolution layer with a tinyMLPthat slides
across the image.
The Block Structure:
1.Spatial Layer:3×3 Conv (Extracts neighbor info)
2.MLP Layer 1:1×1 Conv + ReLU (Feature mixing)
3.MLP Layer 2:1×1 Conv + ReLU (Further abstraction)
The Secret
The 1×1 Conv is just aMatrix Multiplicationperformed on the channel vector
at every pixel (x,y).
George Michailidis Modern Architectures for CNNs 23 / 63
Modern Architectures for CNNs
Network-in-Network
Step 1: The Spatial Foundation
Input: 28×28×160
Operation: 3×3 Convolution
Kernels: 160 filters of size 3×3×160 - assume padding=1, stride=1.
Math:Y i,j = ReLU(Weights∗Input Window + bias)
Result: A 28×28×160 map where each pixel ”sees” its 3×3 neighborhood.
This layer finds local patterns (e.g., ”is there a vertical edge here?”).
George Michailidis Modern Architectures for CNNs 24 / 63
Modern Architectures for CNNs
Network-in-Network
Step 2: The First Local MLP Layer
Operation: 1×1 Convolution (The Linear Transformation)
At every pixel (x,y), we have a vectorv∈R 160.
We multiply it by a Weight MatrixW∈R 160×160.
The ReLU Operator (The Non-Linear Decision):
Equation:Z x,y = max(0,Wv+b)
Dimensions: 28×28×160→28×28×160
This is a Hidden Layer. It does not look at neighbors anymore; it just asks:
”Based on the 160 patterns found in Step 1, does this pixel look like part of a
’Dog’s Ear’?”
George Michailidis Modern Architectures for CNNs 25 / 63
Modern Architectures for CNNs
Network-in-Network
Step 3: The Second Local MLP Layer
Operation: Another 1×1 Convolution + ReLU
Input: Output from Step 2 (28×28×160).
Math:A x,y = max(0,W 2Zx,y +b 2)
Why two 1x1 layers?
1. Without the ReLU in Step 2, two 1×1 convs would collapse into one single
linear layer.
2. With ReLU, we have a Deep MLP that can model extremely complex per-pixel
logic.
This is the Output Layer of the “Mini-Brain” sitting on this pixel.
George Michailidis Modern Architectures for CNNs 26 / 63
Modern Architectures for CNNs
Network-in-Network
The Global Transition: From 160 to 10
How do we get from 28×28×160 to our final 7×7×10 input?
1. Spatial Shrinkage
Between NiN blocks, we apply MaxPooling.
Two rounds of 2×2 pooling (stride 2) shrink the grid:
28×28→14×14→7×7
Channel Evolution (The Features)
Early Blocks: Use 160 filters to find complex textures/parts.
The Final Block: Is configured with10 filters.
Its 1×1 convs act as a per-pixel classifier: ”Based on these 160 features,
does this specific pixel belong to Class 1, Class 2, ..., or Class 10?”
Result
We end up with 10 ”Heatmaps” of size 7×7. Each map represents the
confidence for one specific class across the image grid.
George Michailidis Modern Architectures for CNNs 27 / 63
Modern Architectures for CNNs
Network-in-Network
Step 4: Global Average Pooling (GAP)
Based on pooling the final block hasClasses(e.g., 10).
Input: 7×7×10 (10 feature maps, one for each class).
Operation: Average all pixels in each 7×7 map.
Math: Score c = 1
49
P7
i=1
P7
j=1 Pixeli,j,c
Instead of a Fully Connected layer at the end (like AlexNet/VGG), NiN just asks:
”On average, how much ’Dog’ signal did we find across the whole image?”
George Michailidis Modern Architectures for CNNs 28 / 63
Modern Architectures for CNNs
Network-in-Network
NiN Dimension Cheat Sheet: From Pixels to Probabilities
Stage Operation Input Shape Output Shape Feature Meaning
InputRaw Image – 28×28×3 RGB Pixels
Block 13×3 Conv + 1×1s 28×28×3 28×28×160 Edges / Colors
DownMaxPool (2×2) 28×28×160 14×14×160 Resolution↓
Block 23×3 Conv + 1×1s 14×14×160 14×14×160 Shapes / Parts
DownMaxPool (2×2) 14×14×160 7×7×160 Spatial Grid↓
Final Block1×1 Conv (10 filters) 7×7×160 7×7×10Class Heatmaps
Global PoolGlobal Avg Pool 7×7×10 1×1×10Final Scores
Transformation Logic
Early/Mid Stages:Increasedepth(number of channels) to learn rich features.
Final Stage:Reduce depth to the number of classes (10), then collapse spatial
dimensions (7×7→1×1) using Global Average Pooling.
George Michailidis Modern Architectures for CNNs 29 / 63
Modern Architectures for CNNs
Network-in-Network
Parameter Efficiency: NiN vs. Traditional FC
Let’s compare the Parameter Count for the classification head
(Cin = 160,Classes = 10):
Scenario A: AlexNet/VGG Style (Flattening)
Input:7×7×160 feature maps.
Flattening:Creates a 1D vector of 7840 units.
FC Layer:Connects 7840→10 output units.
Parameters: 7840×10 + 10 =78,410
Scenario B: NiN Style (GAP)
Final1×1Conv:Reduces 160 channels→10 channels.
Parameters: 160×10 + 10 =1,610
GAP Step:Averages each 7×7 map (7×7→1×1).
GAP Parameters:0(it is a fixed math operation).
Summary
NiN provides a≈50x reductionin parameters. By using GAP instead of
Flattening, we force the network to learnglobal conceptsrather thanspecific
pixel positions.
George Michailidis Modern Architectures for CNNs 30 / 63
Modern Architectures for CNNs
Network-in-Network
Summary: Classic CNN vs. NiN
Feature Classic (VGG) NiN
Local Operator Linear Filter + ReLU Micro-MLP (1x1 Convs)
Final Layers Dense (FC) Layers Global Average Pooling
Parameters Very High (in FC) Significantly Lower
Overfitting Risk High Low (Regularized by GAP)
The Legacy of NiN:
1×1 convolutions are the ”glue” of Inception and ResNet.
GAP is now the standard for modern classification heads.
George Michailidis Modern Architectures for CNNs 31 / 63
Modern Architectures for CNNs
Network-in-Network
Motivation: Why 1×1 Convolutions?
Classic Conv:A sliding window that acts as a linear local operator.
The Problem:Classic convs are great at spatial features but weak at
”channel-wise” reasoning.
The NiN Solution:Treat each spatial pixel across all channels as a vector
input to ashared MLP.
The Equivalence:
▶ A 1×1 convolution withC out filters is mathematically identical to aFully
Connected (Dense) layerapplied to each pixel independently.
▶ Dimensions:It transforms depth (C in →C out) while preserving height and
width (H×W).
George Michailidis Modern Architectures for CNNs 32 / 63
Modern Architectures for CNNs
Network-in-Network
The NiN Block: A ”Mini-Brain” for Every Pixel
A NiN block replaces the standard Convolution layer with a tinyMulti-Layer
Perceptron (MLP)that slides across the image.
The Block Architecture:
1.Spatial Layer:3×3 Conv (Extracts neighbor info)
2.MLP Layer 1:1×1 Conv + ReLU (First hidden layer)
3.MLP Layer 2:1×1 Conv + ReLU (Second hidden layer)
Insight for Students
The 3×3 layer ”looks around” to see local context. The 1×1 layers ”think”
about what those combined features mean at that specific spot.
George Michailidis Modern Architectures for CNNs 33 / 63
Modern Architectures for CNNs
Network-in-Network
Step 1: The Spatial Foundation
Input Shape:28×28×160 (Height×Width×Channels)
Operation:3×3Convolution
Settings:160 filters,padding=1, stride=1.
Math:Y= ReLU(Weight∗Input +b)
Output Shape:28×28×160
Pedagogical Note:Because we usedpadding=1, the spatial resolution (28×28)
is preserved. We are just ”sharpening” our features before the MLP kicks in.
George Michailidis Modern Architectures for CNNs 34 / 63
Modern Architectures for CNNs
Network-in-Network
Step 2: The First Local MLP Layer
Operation:1×1Convolution + ReLU
Input:A vectorv∈R 160 at every pixel (x,y).
Linear Step:Multiply by Weight MatrixW 1 ∈R 160×160.
Activation:Z x,y = max(0,W 1v+b 1)
Resulting Shape:28×28×160
Student View:This is exactly like a Hidden Layer in a standard MLP. It allows the
network to learn complex correlationsbetweenthe 160 different feature maps.
George Michailidis Modern Architectures for CNNs 35 / 63
Modern Architectures for CNNs
Network-in-Network
Step 3: Deepening the Logic
Operation: Second1×1Convolution + ReLU
Input:Output from Step 2 (28×28×160).
Activation:A x,y = max(0,W 2Zx,y +b 2)
Why two layers?
One 1×1 conv is just a linear projection.
Adding a second 1×1 layer with a ReLU in between creates aDeep MLP.
This ”Mini-Brain” can now solve much more non-linear problems at each
pixel.
George Michailidis Modern Architectures for CNNs 36 / 63
Modern Architectures for CNNs
Network-in-Network
The Transition: From Features to Classes
A full NiN model stacks several of these blocks. Two things happen as we move
toward the output:
1. Spatial Shrinkage (The Resolution)
Between NiN blocks, we applyMaxPooling(2×2, stride 2).
28×28→14×14→7×7
2. Channel Evolution (The Final Block)
The Special Case:In the very last NiN block, we setC out =Classes(e.g.,
10).
Its 1×1 convs act as a per-pixel classifier.
Final Shape before Output:7×7×10
George Michailidis Modern Architectures for CNNs 37 / 63
Modern Architectures for CNNs
Network-in-Network
Step 4: Global Average Pooling (GAP)
In NiN, we don’t ”Flatten.” We average.
Input:7×7×10 (One ”heatmap” per class).
Operation:Average all 49 pixels in each map.
Math:Score c = 1
49
P7
i=1
P7
j=1 Pixeli,j,c
Philosophy Change
Instead of learning a specific position (e.g., ”The dog is in the top left”), GAP
asks:”How much total ’Dog-ness’ is present in the entire image?”
George Michailidis Modern Architectures for CNNs 38 / 63
Modern Architectures for CNNs
Network-in-Network
Parameter Efficiency: NiN vs. Traditional FC
Let’s compare the Parameter Count for the classification head
(Cin = 160,Classes = 10):
Scenario A: AlexNet/VGG Style (Flattening)
Input: 7×7×160 map.
Flatten to vector of size7,840.
FC Layer: 7,840×10 weights + 10 biases.
Parameters:78,410
Scenario B: NiN Style (1x1 Conv + GAP)
Last 1×1 Conv: 160 channels→10 channels.
Parameters: (1×1×160)×10 + 10 =1,610.
GAP Step:0 parameters.
Summary
NiN is roughly50x more efficientin the classification head, significantly reducing
overfitting.
George Michailidis Modern Architectures for CNNs 39 / 63
Modern Architectures for CNNs
GoogLeNet
Motivation: Moving Beyond VGG
By 2014, the consensus was: Deeper is Better. But how do we go deeper without
the “Computational Explosion”?
Key Question: Should we use 1×1,3×3,or 5×5 filters?
▶ Small kernels (1×1,3×3) capture fine details.
▶ Large kernels (5×5) capture global, sparse features.
GoogLeNet’s Solution: Don’t choose. Use them all in parallel.
Key Innovation:
The Inception Block acts as a ”multi-scale” feature extractor, allowing the
network to decide which size filter is most important for a given feature.
George Michailidis Modern Architectures for CNNs 40 / 63
Modern Architectures for CNNs
GoogLeNet
Anatomy of an Inception Block
An Inception block consists of four parallel paths that are concatenated at the end.
1.Path 1:1×1 Convolution.
2.Path 2:1×1 Conv→3×3 Conv.
3.Path 3:1×1 Conv→5×5 Conv.
4.Path 4:3×3 Max Pool→1×1 Conv.
Note: All paths use padding to ensure the height and width of the output remain
identical for concatenation.
George Michailidis Modern Architectures for CNNs 41 / 63
Modern Architectures for CNNs
GoogLeNet
The 1x1 ”Bottleneck” Strategy
Why are there 1×1 convolutionsbeforethe 3×3 and 5×5 layers?
The Computational Challenge:
5×5 convolutions are expensive. If the input has 192 channels, the
parameter count is massive.
The Solution (Dimensionality Reduction):
We use a 1×1 convolution to ”squish” 192 channels down to (e.g.) 16
channels.
The 5×5 convolution then processes only 16 channels.
Parameter Savings
By adding these ”bottleneck” 1×1 layers, GoogLeNet achieves a22-layer depth
with only7 million parameters(AlexNet had 60 million!).
George Michailidis Modern Architectures for CNNs 42 / 63
Modern Architectures for CNNs
GoogLeNet
GoogLeNet: The Full Pipeline
GoogLeNet is organized into 5 stages (or modules):
Stage 1 & 2:Standard Conv and Max-Pooling (The ”Stem”).
Stage 3, 4 & 5:Stacks of Inception Blocks.
The Head:Like NiN, it usesGlobal Average Pooling (GAP)instead of
heavy FC layers.
George Michailidis Modern Architectures for CNNs 43 / 63
Modern Architectures for CNNs
GoogLeNet
Digression: the Calculations for the “Squish”: 1x1 Convolution
How do we reduce 192 channels to 16? Recall that we treat each pixel in the
feature map as a vector (since we have in this example 192 such maps/channels).
1. The Parameter Count Definition For a layer withC in input channels andC out
output channels:
Params = (Kh ×K w ×C in)×C out + biases
For our “Squish” (1×1 Conv):
Kh = 1,K w = 1 (the spatial footprint is a single pixel).
Cin = 192,C out = 16.
Math: (1×1×192)×16 =3,072weights.
The Matrix Equivalence
At every spatial location (i,j), we perform a matrix-vector multiplication:
yi,j =σ(Wx i,j +b)
Wherex∈R 192,W∈R 16×192, andy∈R 16.
George Michailidis Modern Architectures for CNNs 44 / 63
Modern Architectures for CNNs
GoogLeNet
Why Bottlenecks? A Parameter Comparison
Goal: Apply a 5×5 convolution to 192 channels to get 32 output channels.
Option A: Direct 5x5 Convolution
Math: (5×5×192)×16 =76,800parameters.
Cost:Very heavy; the kernel must process all 192 channels at once.
Option B: 1x1 Bottleneck→5x5 Convolution
1.Reduce:1×1 conv (192→16): (1×1×192)×16 =3,072
2.Process:5×5 conv (16→16): (5×5×16)×16 =6,400
The Efficiency Win
Total Params (Option B):3,072 + 6,400 =9,472
Savings:≈8×fewer parameters for the same spatial result!
Insight: The1×1layer selects the most useful features before the expensive
5×5layer does the spatial work.
George Michailidis Modern Architectures for CNNs 45 / 63
Modern Architectures for CNNs
GoogLeNet
Summary: Why GoogLeNet Won ILSVRC 2014
Feature Impact
Parallel PathsMulti-scale feature extraction.
1x1 BottlenecksReduced complexity and parameter count.
Deep & SparseEfficient use of computing resources.
No Large FC TailMassive reduction in overfitting risk.
Legacy:
The 1×1 bottleneck became a standard design pattern in modern networks
likeResNetandDenseNet.
Proved thatstructural sparsityis superior to brute-force depth.
George Michailidis Modern Architectures for CNNs 46 / 63
Modern Architectures for CNNs
Batch Normalization
Motivation: Why Normalize?
Data Preprocessing: Standardizing input features (µ= 0, σ= 1) makes
optimization much easier.
The Internal Challenge: As a network trains, weights in early layers change.
This shifts the distribution of inputs to later layers (Internal Covariate Shift).
Numerical Stability: Intermediate layer outputs can take widely varying
magnitudes, leading to:
▶ Vanishing gradients (e.g., in Sigmoid/Tanh).
▶ Exploding gradients.
▶ Slow convergence.
Batch Normalization (Ioffe and Szegedy, 2015)
Apply a normalization step inside the network for every mini-batch.
George Michailidis Modern Architectures for CNNs 47 / 63
Modern Architectures for CNNs
Batch Normalization
The Mathematical Formulation
For a mini-batchB={x 1, . . . ,xm}, the BatchNorm transformBN γ,β (xi) is:
1. Batch Mean:µ B = 1
m
Pm
i=1 xi
2. Batch Variance:σ 2
B = 1
m
Pm
i=1(xi −µ B)2 +ϵ
3. Normalize:ˆx i = xi −µB
σB
4. Scale and Shift:y i =γ⊙ˆx i +β
γ(scale) andβ(shift) arelearnable parameters.
ϵ >0 is a small constant to prevent division by zero.
BatchNorm allows the network to recover the original distribution if needed
(γ=σ, β=µ).
George Michailidis Modern Architectures for CNNs 48 / 63
Modern Architectures for CNNs
Batch Normalization
Placement in the Network
BatchNorm is typically applied after the affine transformation (Conv or Fully
Connected) and before the activation function.
1. Fully Connected Layers:
▶ h=ϕ(BN(Wx+b))
▶ Statistics are calculated across all examples in the batch for each feature.
2. Convolutional Layers:
▶ Statistics are calculatedper channel, across all examplesandall spatial
locations (m·p·qelements).
▶ This preserves translation invariance: the sameγandβare applied to all pixels
in a channel.
George Michailidis Modern Architectures for CNNs 49 / 63
Modern Architectures for CNNs
Batch Normalization
Prediction Mode (Inference)
During testing/prediction, we might process onlyone exampleat a time. A single
example has no meaningful batch variance.
Solution: Moving Averages
During training, the layer maintains arunning meanandrunning variance.
These are updated using momentum (typically 0.9 or 0.99):
Running Mean←(1−momentum)·Running Mean + momentum·µ B
Key Result
At test time, the BatchNorm behavior isdeterministicand depends only on the
statistics accumulated during training.
George Michailidis Modern Architectures for CNNs 50 / 63
Modern Architectures for CNNs
Batch Normalization
Why Use BatchNorm?
Faster Convergence:Allows for significantly higher learning rates.
Less Sensitivity:The network is less sensitive to parameter initialization.
Regularization Effect:Because statistics are calculated over mini-batches, it
adds slight noise to the activations, acting as a form of regularization (similar
to Dropout).
Implementation Tip
If using BatchNorm, the bias term (b) in the preceding Conv/FC layer is
redundant because theβparameter in BatchNorm performs the same shifting role.
George Michailidis Modern Architectures for CNNs 51 / 63
Modern Architectures for CNNs
ResNet
The Challenge of Going Deeper
Intuition: Adding more layers should increase a network’s expressive power.
Challenge: Simply stacking layers often leads todegradation: higher training
error as depth increases.
Key Issue: How do we ensure that adding a layer makes a network at least as
good as its shallower version?
Function Classes
If a larger function classFcontains the smaller oneF ′, we are guaranteed that
increasing complexity doesn’t move us away from the optimal solution.
George Michailidis Modern Architectures for CNNs 52 / 63
Modern Architectures for CNNs
ResNet
From Identity to Residuals
If the identity mappingf(x) =xis optimal, it is easier to push a layer to learn
g(x) = 0 than to learn the identity function from scratch.
Regular Block:Attempts to learn the
mappingf(x) directly.
Residual Block:Learns theresidual
mappingg(x) =f(x)−x. The final
output isg(x) +x.
The connection that passesxdirectly to the addition is called aresidual
connection(or shortcut).
This allows gradients to propagate more easily through deep layers.
George Michailidis Modern Architectures for CNNs 53 / 63
Modern Architectures for CNNs
ResNet
Key Elements of a Residual Block
As a working example, consider a standard residual block based on VGG’s 3×3
design:
1. Path A (Weight Path):
▶ 3×3 Convolution
▶ Batch Normalization
▶ ReLU Activation
▶ 3×3 Convolution
▶ Batch Normalization
2. Path B (Shortcut Path):
▶ Identity mapping (if shapes match).
▶ 1×1 Convolution (if we need to change channels or stride).
Final Step:Add Path A + Path B, then apply the final ReLU.
George Michailidis Modern Architectures for CNNs 54 / 63
Modern Architectures for CNNs
ResNet
Handling Dimension Mismatches
The operationg(x) +xrequiresg(x) andxto have the same shape.
Case 1: Same Shape.xis added directly.
Case 2: Different Shape.When we double the channels or halve the
resolution (stride=2):
▶ We use a 1×1convolutionin the shortcut path.
▶ This adjustsxto match the output dimensions of the weight path.
George Michailidis Modern Architectures for CNNs 55 / 63
Modern Architectures for CNNs
ResNet
The Full ResNet-18 Model
ResNet-18 is composed of 5 stages:
Stage Structure Output Size
Stem7×7 Conv, Stride 2, MaxPool 56×56×64
Res-Block 12 Residual Blocks (64 channels) 56×56×64
Res-Block 22 Residual Blocks (128 channels) 28×28×128
Res-Block 32 Residual Blocks (256 channels) 14×14×256
Res-Block 42 Residual Blocks (512 channels) 7×7×512
HeadGlobal Avg Pool, Fully Connected 10 (or 1000)
Note: The first block of each stage (except the first) downsamples the input.
George Michailidis Modern Architectures for CNNs 56 / 63
Modern Architectures for CNNs
ResNet
Why ResNet Had a Big Impact
Training Stability: Allows training of networks with hundreds or thousands of
layers.
Gradient Flow: Shortcuts act as ”highways” for the gradient, mitigating
vanishing gradient problems.
Legacy: Residual connections are now a standard component in nearly all
modern architectures (including Transformers/BERT/GPT).
George Michailidis Modern Architectures for CNNs 57 / 63
Modern Architectures for CNNs
DenseNet
From ResNet to DenseNet
Both architectures use “shortcuts” to improve gradient flow, but they treat the
information differently:
ResNet (Summation):
xℓ =F(x ℓ−1) +x ℓ−1
▶ The learned residualFand the identity are added.
▶ This creates a ”highway” for gradients but combines features into a single
value.
DenseNet (Concatenation):
xℓ =H ℓ([x0,x 1, . . . ,xℓ−1])
▶ Layerℓreceives feature maps from all preceding layers.
▶ Features are concatenated (stacked) along the channel dimension, preserving
the original information.
George Michailidis Modern Architectures for CNNs 58 / 63
Modern Architectures for CNNs
DenseNet
The Dense Block
A Dense Block consists of multiple convolutional layers where each layer’s input is
the concatenation of all previous layers’ outputs.
Growth Rate (k): If each layer produceskfeature maps, then theℓ-th layer
will haveC 0 +k×(ℓ−1) input channels.
Feature Reuse: This architecture is extremely efficient at reusing features, as
early-layer information is directly accessible to the very last layer.
Calculations: The output of layerℓisx ℓ =H ℓ([x0,x 1,· · ·,x ℓ−1]), where [. . .]
denotes concatenation.
George Michailidis Modern Architectures for CNNs 59 / 63
Modern Architectures for CNNs
DenseNet
Transition Layers: Managing Complexity
Because concatenation increases the number of channels rapidly, DenseNet uses
Transition Layers between Dense Blocks to keep the model size manageable.
Components of a Transition Layer:
1. Batch Normalization
2. 1x1 Convolution: Reduces the number of channels (Compression).
3. 2x2 Average Pooling: Halves the spatial dimensions (H×W).
Key Insight: Dense blocks increase depth and channel count; Transition layers
shrink the maps and compress the depth.
George Michailidis Modern Architectures for CNNs 60 / 63
Modern Architectures for CNNs
DenseNet
Global Architecture
DenseNet follows a similar modular structure to ResNet but swaps Residual
Blocks for Dense Blocks.
Component Function
Stem7×7 Conv + MaxPool (Initial downsampling).
Dense Block 1Sequence ofkconvolutions with concatenation.
Transition 11×1 Conv + 2×2 AvgPool.
Dense Block 2Further feature extraction and reuse.
......
Classification HeadGlobal Avg Pool + Fully Connected Layer.
George Michailidis Modern Architectures for CNNs 61 / 63
Modern Architectures for CNNs
DenseNet
Why DenseNet?
Strong Gradient Flow: Every layer has a direct ”shortcut” to the loss
function via the concatenated paths.
Parameter Efficiency: Because features are reused, DenseNet often requires
fewer parametersthan ResNet for similar accuracy.
Bottleneck Layers: Modern implementations use 1×1 convswithinthe dense
block (DenseNet-B) to reduce input channels before the 3×3 conv.
George Michailidis Modern Architectures for CNNs 62 / 63
Modern Architectures for CNNs
DenseNet
Accessing Architectures: Hugging Face & Libraries
Modern CNNs are rarely written from scratch in production. Instead, we use
Model Hubs and specialized libraries.
The Hugging Face Hub: A central repository for weights and configurations
for thousands of models (ResNet, DenseNet, ConvNeXt).
timm (PyTorch Image Models): The ”gold standard” library integrated with
Hugging Face for Computer Vision.
Loading a Model in Python:
timm.create model(’resnet50’, pretrained=True)
timm.create model(’densenet121’, pretrained=True)
George Michailidis Modern Architectures for CNNs 63 / 63
