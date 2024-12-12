
# GradViT: Gradient-Driven Growing Vision Transformer in Fixed-Point

**Author - Sumanth Prabhu**        
**Department of Electrical and Computer Engineering**  
University of Illinois at Urbana-Champaign  
Urbana, IL 61801

Email: [prabhu5@illinois.edu](mailto:prabhu5@illinois.edu)

## Project Overview
GradViT (Gradient-Driven Growing Vision Transformer in Fixed-Point) aims to develop an innovative Vision Transformer (ViT) architecture that operates entirely in fixed-point arithmetic and can dynamically grow based on loss and gradient information. This project addresses two critical challenges in edge AI:

- **Efficient Fixed-Point Models**: Designing models suitable for edge devices without floating-point units.
- **Dynamic Model Adaptation**: Enabling models to adapt their complexity dynamically based on task difficulty and available resources.

By combining fixed-point arithmetic with a gradient-driven growth mechanism, GradViT promises a highly adaptable and efficient ViT architecture for edge devices. This approach could advance on-device AI, enabling more powerful and flexible models for applications such as autonomous vehicles, smart cameras, and IoT devices.

---

## Methodology

### 1. Fixed-Point ViT Base Architecture
- **Objective**: Implement a basic Vision Transformer architecture using only fixed-point operations.
- **Key Tasks**:
  - Develop fixed-point versions of key operations

### 2. Dynamic Growth Mechanism & Gradient-Driven Adaptation
- **Objective**: Enable the ViT to grow dynamically based on gradient and loss information.
- **Key Tasks**:
  - Design a gradient-based growth trigger that monitors:
    - Loss magnitudes
    - Gradient flow statistics
  - Implement growth operations such as:
    - Adding transformer layers
    - Expanding hidden dimensions
    - Increasing attention heads
  - Develop a method to analyze gradient flow to determine where growth is most beneficial.
  - Implement mechanisms to adjust the fixed-point representation dynamically based on gradient statistics.

---

## Applications
GradViT is designed for edge AI applications, including:
- **Autonomous Vehicles**: Enabling efficient and scalable perception models.
- **Smart Cameras**: Improving on-device image and video analysis.
- **IoT Devices**: Empowering resource-constrained devices with adaptable AI models.

---
## **Utility Functions**

### `fixed_point_quantize`
Quantizes a tensor to a fixed-point representation:
- **Input**: Tensor and a scaling factor (`scale`).
- **Process**:
  - Multiplies the tensor by the scaling factor.
  - Rounds the result to the nearest integer and clamps it to a 8-bit range.
  - Rescales the clamped result to simulate fixed-point behavior.
- **Output**: A fixed-point quantized tensor.

---

## **Components**

### **1. FixedPointMLP**
A fixed-point implementation of a Multi-Layer Perceptron (MLP).

- Defines two fully connected layers (`fc1` and `fc2`) with configurable dimensions.
- Accepts a scaling factor (`scale`) for quantization.

#### **Forward Pass**:
1. Applies the first fully connected layer and quantizes the output.
2. Applies a ReLU activation.
3. Applies the second fully connected layer and quantizes the result.

---

### **2. FixedPointAttention**
Implements self-attention with fixed-point arithmetic.

- Includes a linear layer (`qkv`) to compute queries, keys, and values.
- Includes an output projection layer (`out_proj`).
- Accepts the number of attention heads and the scaling factor (`scale`).

#### **Forward Pass**:
1. Splits the input into query (`q`), key (`k`), and value (`v`) vectors.
2. Reshapes them for multi-head attention.
3. Computes scaled dot-product attention and quantizes the attention scores.
4. Projects the context back to the original embedding dimension and quantizes the result.

---

### **3. FixedPointTransformerEncoderLayer**
Represents a single encoder layer in the Vision Transformer.

- Contains a `FixedPointAttention` module for self-attention.
- Includes an MLP implemented with `FixedPointMLP`.
- Uses LayerNorm layers for normalization.

#### **Forward Pass**:
1. Applies LayerNorm and self-attention, then quantizes the attention output.
2. Adds the residual connection and quantizes the result.
3. Applies LayerNorm and the MLP, then quantizes the output.
4. Adds another residual connection and quantizes the final result.

---

### **4. FixedPointViT**
Implements the Vision Transformer (ViT) architecture with fixed-point arithmetic.

- Parameters:
  - **Image size**: Defines input dimensions.
  - **Patch size**: Determines patch extraction size.
  - **Number of classes**: For classification tasks.
  - **Embedding dimensions**, **depth**, **attention heads**, **hidden dimensions**, and **scale**.
- Components:
  - **Patch embedding**: Flattens image patches into vectors and embeds them.
  - **Positional embeddings**: Adds position information to patches.
  - **Transformer Encoder**: A list of `FixedPointTransformerEncoderLayer` modules.
  - **Classification Head**: A linear layer for final classification.

#### **Forward Pass**:
1. Extracts and flattens patches from the input image.
2. Embeds patches and adds positional embeddings.
3. Passes embeddings through the Transformer encoder layers.
4. Applies global average pooling on the output.
5. Passes the result through the classification head.

---
## Growth Criteria

The model checks the following conditions before triggering growth:

1. **Epoch-Based Trigger**:
   - Growth is only allowed after a certain number of initial epochs .
   - Growth occurs periodically, as determined by the parameter `grow_every_n_epochs`.

2. **Gradient Norm Threshold**:
   - The average gradient norm (`avg_grad_norm`) must exceed a predefined threshold (`grad_threshold`), indicating that the model may need additional capacity.

3. **Loss Threshold**:
   - The epoch loss (`epoch_loss`) must exceed a defined threshold (`loss_threshold`), suggesting the current architecture struggles with task complexity.

---

## Growth Implementation

**When the growth criteria are met, the following steps occur**:
- Create a New Transformer Encoder Layer:
- Weights of the new layer are initialized with a normal distribution:
- Add the Layer to the Model

---

## Contact
For questions or collaboration, feel free to reach out to me @prabhu5@illinois.edu!
