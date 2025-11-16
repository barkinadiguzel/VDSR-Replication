# 🔦 VDSR PyTorch Implementation

This repository contains a replication of **VDSR (Very Deep Super-Resolution network)** using PyTorch. The goal is to reproduce the **VDSR architecture** for single image super-resolution (SISR) tasks.

- Architecture follows: **ConvInput → [ConvBlock × num_blocks] → ConvOutput → ResidualMerge**  
**Paper**: [Accurate Image Super-Resolution Using Very Deep Convolutional Networks (CVPR 2016)](https://arxiv.org/abs/1511.04587)

> 🛠️ Users may need to adjust the code slightly for different input channels or custom super-resolution setups.

---

## 🖼 Overview – VDSR Architecture

![VDSR Overview](images/figuremix.jpg)

- Figures in the image:
  - **Figure 1:** Input ILR image
  - **Figure 2:** ConvBlocks and feature extraction
  - **Figure 5:** Residual added to input for final HR output

> - 🐙VDSR takes an **interpolated low-resolution (ILR) image** as input and predicts the **residual (high-frequency details)**.
> - 🐙The network consists of **ConvInput → multiple ConvBlocks → ConvOutput → ResidualMerge**, forming a very deep CNN (up to 20 layers).
> - 🐙**Residual learning**: predicted residual is added back to ILR to get the final **high-resolution (HR) image**.
> - 🐙**Deep architecture** increases the **receptive field**, allowing the network to use more context from neighboring pixels to improve super-resolution accuracy.
---

## 🏗 Project Structure

```bash
VDSR-Replication/
│
├── src/
│   ├── layers/
│   │   ├── conv_input.py         # First conv layer (input)
│   │   ├── conv_block.py         # 3x3 Conv + ReLU (repeated num_blocks times)
│   │   ├── conv_output.py        # Last conv layer for residual prediction
│   │   ├── residual_merge.py     # Adds predicted residual to ILR input
│   │   └── pad.py                # Zero-padding helper
│   │
│   ├── model/
│   │   └── vdsr.py               # ConvInput → [ConvBlock × num_blocks] → ConvOutput → ResidualMerge
│   │
│   └── config.py                 # Hyperparameters (num_blocks, channels, padding, etc.)
│
├── requirements.txt
└── README.md
```
---

## 🔗 Feedback

For questions or feedback, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
