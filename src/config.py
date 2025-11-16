# VDSR Configuration

# Network depth
num_blocks = 20         # 20 weight layers as in the paper

# Input image
input_channels = 1      # ILR: single-channel grayscale or 3 for RGB
input_size = None       # Flexible, depends on the image

# Conv layers
num_filters = 64        # All hidden conv layers have 64 filters
filter_size = 3         # 3x3 kernel
padding = 1             # Zero-padding to keep feature map size

# Residual learning
residual = True         # Predict residual image (HR - ILR)

# Training hyperparameters (optional here, can be in a separate train config)
learning_rate = 0.1     # Extremely high initial LR
clip_gradient = True    # Enable adjustable gradient clipping
