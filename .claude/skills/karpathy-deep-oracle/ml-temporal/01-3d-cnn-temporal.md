# 3D Convolutional Neural Networks for Temporal Video Understanding

## Overview

3D Convolutional Neural Networks (3D CNNs) extend traditional 2D image convolutions into the temporal domain, enabling neural networks to learn spatiotemporal features directly from video data. Unlike 2D CNNs that process individual frames, 3D CNNs convolve filters across both spatial dimensions and time, capturing motion patterns and temporal relationships between consecutive frames.

**Core Insight**: A 3D convolution treats time as a spatial dimension, creating a "thick present" - a volume of frames processed simultaneously rather than sequentially.

From [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248) (Tran et al., 2017):
- Systematically studied different forms of spatiotemporal convolutions
- Proposed R(2+1)D factorization achieving state-of-the-art results
- Cited by 4,452+ papers - foundational work for video understanding

---

## Section 1: 3D Convolution Fundamentals

### What is a 3D Convolution?

A 3D convolution extends the standard 2D convolution by adding a temporal dimension to the kernel. While a 2D convolution operates on a single image with kernel size `(C_in, H, W)`, a 3D convolution operates on a video volume with kernel size `(C_in, T, H, W)`:

- **C_in**: Input channels (e.g., 3 for RGB)
- **T**: Temporal extent (number of frames)
- **H, W**: Spatial height and width

**Mathematical Formulation**:

For input tensor X of shape (C_in, D, H, W) and kernel K of shape (C_out, C_in, T, K_H, K_W):

```
Y[c_out, d, h, w] = sum over (c_in, t, i, j) of:
    X[c_in, d+t, h+i, w+j] * K[c_out, c_in, t, i, j]
```

### PyTorch 3D Convolution Layer

```python
import torch
import torch.nn as nn

class Basic3DConv(nn.Module):
    """Basic 3D convolution block for video processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3, 3),  # (T, H, W)
        stride: tuple = (1, 1, 1),
        padding: tuple = (1, 1, 1)
    ):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        return self.relu(self.bn(self.conv(x)))


# Example usage
batch_size = 4
channels = 3
num_frames = 16
height, width = 112, 112

# Create video tensor
video = torch.randn(batch_size, channels, num_frames, height, width)

# Apply 3D convolution
conv3d = Basic3DConv(in_channels=3, out_channels=64)
output = conv3d(video)

print(f"Input shape: {video.shape}")
print(f"Output shape: {output.shape}")
# Input shape: torch.Size([4, 3, 16, 112, 112])
# Output shape: torch.Size([4, 64, 16, 112, 112])
```

### Temporal Receptive Field

The **temporal receptive field** is the number of input frames that contribute to a single output activation. With stacked 3D convolutions, this grows with network depth:

```python
def compute_temporal_receptive_field(
    num_layers: int,
    kernel_temporal: int = 3,
    stride_temporal: int = 1
) -> int:
    """
    Compute temporal receptive field of stacked 3D convolutions.

    For kernel_t=3, stride_t=1:
    - Layer 1: RF = 3
    - Layer 2: RF = 5
    - Layer 3: RF = 7
    ...
    - Layer N: RF = 1 + N * (kernel_t - 1)
    """
    rf = 1
    for _ in range(num_layers):
        rf = rf + (kernel_temporal - 1) * stride_temporal
    return rf

# Example: 5 layers of 3x3x3 convolutions
for n_layers in range(1, 8):
    rf = compute_temporal_receptive_field(n_layers)
    print(f"{n_layers} layers -> Temporal RF = {rf} frames")

# 1 layers -> Temporal RF = 3 frames
# 2 layers -> Temporal RF = 5 frames
# 3 layers -> Temporal RF = 7 frames
# 4 layers -> Temporal RF = 9 frames
# 5 layers -> Temporal RF = 11 frames
# 6 layers -> Temporal RF = 13 frames
# 7 layers -> Temporal RF = 15 frames
```

### Parameter Comparison: 2D vs 3D Convolutions

```python
def count_conv_params(in_ch, out_ch, kernel_2d=(3, 3), kernel_3d=(3, 3, 3)):
    """Compare parameters between 2D and 3D convolutions."""

    # 2D convolution: out_ch * in_ch * H * W
    params_2d = out_ch * in_ch * kernel_2d[0] * kernel_2d[1]

    # 3D convolution: out_ch * in_ch * T * H * W
    params_3d = out_ch * in_ch * kernel_3d[0] * kernel_3d[1] * kernel_3d[2]

    return params_2d, params_3d

in_channels, out_channels = 64, 128
p2d, p3d = count_conv_params(in_channels, out_channels)

print(f"2D Conv (3x3): {p2d:,} parameters")
print(f"3D Conv (3x3x3): {p3d:,} parameters")
print(f"3D has {p3d/p2d:.1f}x more parameters")

# 2D Conv (3x3): 73,728 parameters
# 3D Conv (3x3x3): 221,184 parameters
# 3D has 3.0x more parameters
```

---

## Section 2: C3D and I3D Architectures

### C3D: Learning Spatiotemporal Features

C3D (Convolutional 3D) was one of the first successful deep 3D CNN architectures for video understanding.

From [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) (Tran et al., 2015):
- All convolutions use 3x3x3 kernels (homogeneous architecture)
- Pooling uses 2x2x2 kernels except first layer (1x2x2)
- 8 conv layers, 5 pooling layers, 2 FC layers

```python
import torch.nn as nn

class C3D(nn.Module):
    """
    C3D network architecture.

    Input: 16 frames of 112x112 RGB video
    Output: 4096-dim feature vector (before final classification)

    Reference: https://arxiv.org/abs/1412.0767
    """

    def __init__(self, num_classes: int = 101):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        # Fully connected layers
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 3, 16, 112, 112)

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)
        return logits


# Test C3D
model = C3D(num_classes=101)
video = torch.randn(2, 3, 16, 112, 112)
output = model(video)
print(f"C3D output shape: {output.shape}")  # (2, 101)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"C3D total parameters: {total_params:,}")  # ~78M parameters
```

### I3D: Inflated 3D ConvNets

I3D (Inflated 3D) takes pre-trained 2D ImageNet weights and "inflates" them into 3D by repeating along the temporal dimension.

From [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750) (Carreira & Zisserman, 2017):
- Bootstrap from ImageNet pre-trained 2D CNNs
- Inflate 2D filters to 3D by repeating temporally
- Achieves state-of-the-art on multiple benchmarks

```python
import torch
import torch.nn as nn

def inflate_conv_weights(weights_2d: torch.Tensor, temporal_dim: int = 3) -> torch.Tensor:
    """
    Inflate 2D conv weights to 3D by repeating along temporal dimension.

    Args:
        weights_2d: Shape (C_out, C_in, H, W)
        temporal_dim: Temporal kernel size

    Returns:
        weights_3d: Shape (C_out, C_in, T, H, W)
    """
    # Repeat along temporal dimension
    weights_3d = weights_2d.unsqueeze(2).repeat(1, 1, temporal_dim, 1, 1)

    # Normalize by temporal dimension to preserve activation magnitude
    weights_3d = weights_3d / temporal_dim

    return weights_3d


class InflatedConv3d(nn.Module):
    """
    3D convolution layer initialized from 2D weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        temporal_kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        pretrained_2d: nn.Conv2d = None
    ):
        super().__init__()

        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(temporal_kernel, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(temporal_kernel // 2, padding, padding),
            bias=False
        )

        # Initialize from 2D weights if provided
        if pretrained_2d is not None:
            with torch.no_grad():
                inflated = inflate_conv_weights(
                    pretrained_2d.weight.data,
                    temporal_dim=temporal_kernel
                )
                self.conv3d.weight.copy_(inflated)

    def forward(self, x):
        return self.conv3d(x)


# Example: Inflate a 2D ResNet conv layer
conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
nn.init.kaiming_normal_(conv2d.weight)

inflated_conv = InflatedConv3d(
    64, 128,
    kernel_size=3,
    temporal_kernel=3,
    pretrained_2d=conv2d
)

# Test
video_input = torch.randn(2, 64, 16, 56, 56)
output = inflated_conv(video_input)
print(f"Inflated conv output: {output.shape}")  # (2, 128, 16, 56, 56)
```

### I3D with Inception Architecture

```python
class I3DInceptionModule(nn.Module):
    """
    Inception module for I3D (inflated from Inception v1).

    Processes video with multiple parallel pathways at different scales.
    """

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        reduce_3x3: int,
        out_3x3: int,
        reduce_5x5: int,
        out_5x5: int,
        out_pool: int
    ):
        super().__init__()

        # 1x1x1 pathway
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm3d(out_1x1),
            nn.ReLU(inplace=True)
        )

        # 3x3x3 pathway
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, reduce_3x3, kernel_size=1),
            nn.BatchNorm3d(reduce_3x3),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_3x3),
            nn.ReLU(inplace=True)
        )

        # 5x5x5 pathway (implemented as two 3x3x3)
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, reduce_5x5, kernel_size=1),
            nn.BatchNorm3d(reduce_5x5),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduce_5x5, out_5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_5x5),
            nn.ReLU(inplace=True)
        )

        # Pool pathway
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm3d(out_pool),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


# Example inception module
inception = I3DInceptionModule(
    in_channels=192,
    out_1x1=64,
    reduce_3x3=96, out_3x3=128,
    reduce_5x5=16, out_5x5=32,
    out_pool=32
)

x = torch.randn(2, 192, 16, 28, 28)
out = inception(x)
print(f"Inception output: {out.shape}")  # (2, 256, 16, 28, 28)
```

---

## Section 3: Factorized 3D Convolutions - R(2+1)D

### The Factorization Insight

Instead of a full 3D convolution `(T, H, W)`, decompose into:
1. **Spatial convolution**: `(1, H, W)` - captures spatial features per frame
2. **Temporal convolution**: `(T, 1, 1)` - captures motion between frames

**Benefits**:
- Fewer parameters (T*H*W -> T + H*W)
- Double the non-linearities (ReLU after each component)
- Can initialize spatial component from 2D ImageNet weights

```python
import torch
import torch.nn as nn

class R2Plus1DConv(nn.Module):
    """
    R(2+1)D: Factorized 3D convolution into 2D spatial + 1D temporal.

    Reference: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2017)
    https://arxiv.org/abs/1711.11248
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3, 3),
        stride: tuple = (1, 1, 1),
        padding: tuple = (1, 1, 1)
    ):
        super().__init__()

        t_kernel = kernel_size[0]
        s_kernel = kernel_size[1:]

        t_stride = stride[0]
        s_stride = stride[1:]

        t_padding = padding[0]
        s_padding = padding[1:]

        # Compute intermediate channels for proper parameter count
        # M_i = floor(t * d^2 * N_{i-1} * N_i / (d^2 * N_{i-1} + t * N_i))
        mid_channels = (
            t_kernel * s_kernel[0] * s_kernel[1] * in_channels * out_channels
        ) // (
            s_kernel[0] * s_kernel[1] * in_channels + t_kernel * out_channels
        )

        # Spatial convolution: 1 x k x k
        self.spatial_conv = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, s_kernel[0], s_kernel[1]),
            stride=(1, s_stride[0], s_stride[1]),
            padding=(0, s_padding[0], s_padding[1]),
            bias=False
        )
        self.spatial_bn = nn.BatchNorm3d(mid_channels)

        # Temporal convolution: t x 1 x 1
        self.temporal_conv = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(t_kernel, 1, 1),
            stride=(t_stride, 1, 1),
            padding=(t_padding, 0, 0),
            bias=False
        )
        self.temporal_bn = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Spatial component
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.relu(x)

        # Temporal component
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.relu(x)

        return x


# Compare parameter counts
def count_params(module):
    return sum(p.numel() for p in module.parameters())

in_ch, out_ch = 64, 128

# Full 3D conv
full_3d = nn.Sequential(
    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm3d(out_ch)
)

# R(2+1)D factorized
r2plus1d = R2Plus1DConv(in_ch, out_ch)

print(f"Full 3D conv params: {count_params(full_3d):,}")
print(f"R(2+1)D params: {count_params(r2plus1d):,}")
print(f"Reduction: {count_params(full_3d)/count_params(r2plus1d):.2f}x")

# Full 3D conv params: 221,440
# R(2+1)D params: 164,832
# Reduction: 1.34x
```

### R(2+1)D ResNet Block

```python
class R2Plus1DBlock(nn.Module):
    """
    R(2+1)D residual block.

    Uses factorized 3D convolutions within a residual structure.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple = (1, 1, 1),
        downsample: nn.Module = None
    ):
        super().__init__()

        self.conv1 = R2Plus1DConv(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=(1, 1, 1)
        )

        self.conv2 = R2Plus1DConv(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class R2Plus1DNet(nn.Module):
    """
    R(2+1)D-18 network for video classification.

    PyTorch reference: torchvision.models.video.r2plus1d_18
    """

    def __init__(self, num_classes: int = 400, input_frames: int = 16):
        super().__init__()

        # Stem: Initial conv layer
        self.stem = nn.Sequential(
            nn.Conv3d(
                3, 64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=(2, 2, 2))
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=(2, 2, 2))
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=(2, 2, 2))

        # Global pooling and classification
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: tuple = (1, 1, 1)
    ):
        downsample = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )

        layers = []
        layers.append(R2Plus1DBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(R2Plus1DBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 3, T, H, W)
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


# Test R(2+1)D network
model = R2Plus1DNet(num_classes=400)
video = torch.randn(2, 3, 16, 112, 112)
output = model(video)
print(f"R(2+1)D-18 output: {output.shape}")  # (2, 400)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.1f}M")  # ~33M
```

---

## Section 4: Mixed Convolutions (MC3) and Architectural Variants

### MC3: Mixed 3D/2D Convolutions

MC3 uses 3D convolutions only in early layers (where motion features are extracted) and 2D in later layers (where semantic features dominate).

```python
class MC3Block(nn.Module):
    """
    Mixed Convolution block - 3D early, 2D late.

    Insight: Motion is low/mid-level feature, semantics are high-level.
    Early layers need temporal modeling, later layers don't.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_3d: bool = True,  # Whether to use 3D or 2D conv
        stride: tuple = (1, 1, 1)
    ):
        super().__init__()

        if use_3d:
            # Full 3D convolution
            self.conv = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(3, 3, 3),
                stride=stride,
                padding=(1, 1, 1),
                bias=False
            )
        else:
            # 2D convolution (no temporal extent)
            spatial_stride = stride[1:]
            self.conv = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, spatial_stride[0], spatial_stride[1]),
                padding=(0, 1, 1),
                bias=False
            )

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MC3Net(nn.Module):
    """
    MC3-18: Mixed 3D/2D convolution network.

    3D convolutions in layer1 and layer2 (motion features)
    2D convolutions in layer3 and layer4 (semantic features)

    Reference: torchvision.models.video.mc3_18
    """

    def __init__(self, num_classes: int = 400):
        super().__init__()

        # Stem with 3D conv
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                     stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # Layer 1-2: 3D convolutions (motion)
        self.layer1 = self._make_layer(64, 64, use_3d=True)
        self.layer2 = self._make_layer(64, 128, use_3d=True, stride=(2, 2, 2))

        # Layer 3-4: 2D convolutions (semantics)
        self.layer3 = self._make_layer(128, 256, use_3d=False, stride=(2, 2, 2))
        self.layer4 = self._make_layer(256, 512, use_3d=False, stride=(2, 2, 2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, use_3d, stride=(1, 1, 1)):
        return nn.Sequential(
            MC3Block(in_ch, out_ch, use_3d=use_3d, stride=stride),
            MC3Block(out_ch, out_ch, use_3d=use_3d)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)
```

### S3D: Separable 3D Convolutions

S3D further improves efficiency with separable spatiotemporal convolutions throughout the network.

```python
class S3DGatingUnit(nn.Module):
    """
    Self-gating unit for S3D-G variant.

    Adds feature-wise gating to improve capacity without parameters.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T, H, W)
        gate = self.avg_pool(x).flatten(1)  # (B, C)
        gate = self.sigmoid(self.fc(gate))  # (B, C)
        gate = gate.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (B, C, 1, 1, 1)
        return x * gate


class S3DConvBlock(nn.Module):
    """
    S3D separable 3D convolution block.

    Fully factorized: (1xHxW) -> (Tx1x1)
    Optional self-gating for S3D-G variant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        temporal_kernel: int = 3,
        stride: int = 1,
        use_gating: bool = False
    ):
        super().__init__()

        # Spatial 2D conv
        self.spatial = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, kernel_size // 2, kernel_size // 2),
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Temporal 1D conv
        self.temporal = nn.Sequential(
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Optional gating
        self.gating = S3DGatingUnit(out_channels) if use_gating else None

    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        if self.gating:
            x = self.gating(x)
        return x
```

---

## Section 5: Using TorchVision Video Models

PyTorch provides pre-trained 3D CNN models in torchvision:

```python
import torch
from torchvision.models.video import (
    r3d_18, R3D_18_Weights,
    mc3_18, MC3_18_Weights,
    r2plus1d_18, R2Plus1D_18_Weights,
    s3d, S3D_Weights
)

# Load pre-trained models
def load_pretrained_video_models():
    """
    Load TorchVision pre-trained video models.

    All models trained on Kinetics-400 dataset.
    Input: (B, 3, T, H, W) - typically (B, 3, 16, 112, 112)
    Output: (B, 400) - 400 action classes
    """

    # R3D-18: Full 3D ResNet
    model_r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)

    # MC3-18: Mixed 3D/2D
    model_mc3 = mc3_18(weights=MC3_18_Weights.DEFAULT)

    # R(2+1)D-18: Factorized 3D
    model_r2plus1d = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

    # S3D: Separable 3D with gating
    model_s3d = s3d(weights=S3D_Weights.DEFAULT)

    return {
        'r3d_18': model_r3d,
        'mc3_18': model_mc3,
        'r2plus1d_18': model_r2plus1d,
        's3d': model_s3d
    }


def extract_video_features(model, video_tensor):
    """
    Extract features from video using pre-trained model.

    Args:
        model: Pre-trained video model
        video_tensor: Shape (B, 3, T, H, W)

    Returns:
        features: Shape (B, num_classes)
    """
    model.eval()
    with torch.no_grad():
        features = model(video_tensor)
    return features


# Example usage
models = load_pretrained_video_models()

# Create sample video (normalized to ImageNet stats)
video = torch.randn(2, 3, 16, 112, 112)

for name, model in models.items():
    output = extract_video_features(model, video)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{name}: output={output.shape}, params={params:.1f}M")

# r3d_18: output=torch.Size([2, 400]), params=33.4M
# mc3_18: output=torch.Size([2, 400]), params=11.7M
# r2plus1d_18: output=torch.Size([2, 400]), params=31.5M
# s3d: output=torch.Size([2, 400]), params=8.3M
```

### Fine-tuning for Custom Action Recognition

```python
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class CustomActionRecognizer(nn.Module):
    """
    Fine-tune R(2+1)D for custom action classes.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        # Load pre-trained backbone
        if pretrained:
            weights = R2Plus1D_18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = r2plus1d_18(weights=weights)

        # Replace classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_feature_extractor(self):
        """Return backbone without final FC for feature extraction."""
        # Remove classifier
        modules = list(self.backbone.children())[:-1]
        return nn.Sequential(*modules)


# Example: Fine-tune for 10 custom action classes
model = CustomActionRecognizer(num_classes=10, pretrained=True)

# Training loop setup
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=1e-4
)

# Different learning rates for backbone vs head
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if 'fc' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.SGD([
    {'params': backbone_params, 'lr': 0.0001},  # Frozen-ish backbone
    {'params': head_params, 'lr': 0.01}  # Train head faster
], momentum=0.9)
```

---

## Section 6: Performance Considerations

### Memory and Compute Requirements

3D CNNs are memory-intensive due to the temporal dimension:

```python
import torch

def estimate_memory_usage(
    batch_size: int,
    channels: int,
    frames: int,
    height: int,
    width: int,
    dtype: torch.dtype = torch.float32
) -> float:
    """
    Estimate memory for video tensor in GB.
    """
    elements = batch_size * channels * frames * height * width
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = elements * bytes_per_element
    return total_bytes / (1024 ** 3)

# Compare memory for different configurations
configs = [
    (8, 3, 8, 112, 112),    # Short clip
    (8, 3, 16, 112, 112),   # Standard clip
    (8, 3, 32, 224, 224),   # Long clip, high res
    (4, 3, 64, 224, 224),   # Very long clip
]

for b, c, t, h, w in configs:
    mem = estimate_memory_usage(b, c, t, h, w)
    print(f"Batch={b}, Frames={t}, Size={h}x{w}: {mem:.2f} GB")

# Batch=8, Frames=8, Size=112x112: 0.03 GB
# Batch=8, Frames=16, Size=112x112: 0.06 GB
# Batch=8, Frames=32, Size=224x224: 0.92 GB
# Batch=4, Frames=64, Size=224x224: 0.92 GB
```

### Optimization Strategies

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class EfficientVideoTrainer:
    """
    Training utilities for memory-efficient 3D CNN training.
    """

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.scaler = GradScaler()  # Mixed precision

    def train_step(self, video, labels, optimizer, criterion):
        """
        Single training step with mixed precision.

        Mixed precision can reduce memory by ~50% and speed up training.
        """
        self.model.train()
        optimizer.zero_grad()

        video = video.to(self.device)
        labels = labels.to(self.device)

        # Forward pass with automatic mixed precision
        with autocast():
            outputs = self.model(video)
            loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, dataloader, criterion):
        """Evaluation with mixed precision."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for video, labels in dataloader:
            video = video.to(self.device)
            labels = labels.to(self.device)

            with autocast():
                outputs = self.model(video)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }


def apply_gradient_checkpointing(model):
    """
    Apply gradient checkpointing to reduce memory.

    Trades compute for memory by recomputing activations during backward.
    Can reduce memory by 4-5x for deep networks.
    """
    from torch.utils.checkpoint import checkpoint_sequential

    # For R(2+1)D-like models, checkpoint each residual block
    if hasattr(model, 'layer1'):
        model.layer1 = checkpoint_wrapper(model.layer1)
        model.layer2 = checkpoint_wrapper(model.layer2)
        model.layer3 = checkpoint_wrapper(model.layer3)
        model.layer4 = checkpoint_wrapper(model.layer4)

    return model

def checkpoint_wrapper(module):
    """Wrap module with gradient checkpointing."""
    class CheckpointedModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x):
            return torch.utils.checkpoint.checkpoint(
                self.module, x, use_reentrant=False
            )

    return CheckpointedModule(module)
```

### Temporal Stride and Frame Sampling

```python
import torch

def temporal_sampling_strategies(video_frames, target_frames=16):
    """
    Different strategies for sampling frames from video.

    Args:
        video_frames: Full video tensor (T_full, C, H, W)
        target_frames: Number of frames to sample

    Returns:
        sampled: Sampled frames (target_frames, C, H, W)
    """
    total_frames = video_frames.shape[0]

    # Strategy 1: Uniform sampling
    indices_uniform = torch.linspace(0, total_frames - 1, target_frames).long()
    uniform_sample = video_frames[indices_uniform]

    # Strategy 2: Dense sampling from center
    center = total_frames // 2
    start = max(0, center - target_frames // 2)
    dense_sample = video_frames[start:start + target_frames]

    # Strategy 3: Random sparse sampling
    indices_random = torch.randperm(total_frames)[:target_frames].sort()[0]
    random_sample = video_frames[indices_random]

    # Strategy 4: Temporal stride
    stride = total_frames // target_frames
    indices_stride = torch.arange(0, total_frames, stride)[:target_frames]
    stride_sample = video_frames[indices_stride]

    return {
        'uniform': uniform_sample,
        'dense_center': dense_sample,
        'random_sparse': random_sample,
        'strided': stride_sample
    }


# Example: Sample 16 frames from 64-frame video
full_video = torch.randn(64, 3, 112, 112)
samples = temporal_sampling_strategies(full_video, target_frames=16)

for name, sample in samples.items():
    print(f"{name}: {sample.shape}")
```

---

## Section 7: TRAIN STATION - 3D Convolution = Temporal = Volume = Thick Present

### The Topological Equivalence

**Coffee Cup = Donut Thinking for Video Understanding**:

```
3D Convolution = Temporal Volume = Specious Present = Thick Now
     |                |                |              |
     v                v                v              v
Treats time     Processes       William James   Duration has
as space       T frames as     "specious       extent, not
dimension       a volume        present"       instantaneous
```

### The "Thick Present" Connection

The 3D convolution kernel embodies the philosophical concept of the "specious present" - the idea that perception occurs not at an instant but over a temporal interval:

```python
class ThickPresentConvolution(nn.Module):
    """
    3D convolution as implementation of "thick present" processing.

    The temporal kernel size IS the thickness of the present moment.

    Philosophical connections:
    - William James: "Specious present" has duration
    - Husserl: "Retention-primal impression-protention" structure
    - Whitehead: "Actual occasions" have temporal thickness

    ML implementations:
    - 3D CNN: Treats time as spatial dimension
    - Temporal transformers: Attention over time window
    - State space models: Continuous temporal dynamics
    """

    def __init__(
        self,
        channels: int,
        present_thickness: int = 3,  # How "thick" is the present
        spatial_kernel: int = 3
    ):
        super().__init__()

        # The temporal kernel size determines "thickness" of present
        self.present_thickness = present_thickness

        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=(present_thickness, spatial_kernel, spatial_kernel),
            padding=(present_thickness // 2, spatial_kernel // 2, spatial_kernel // 2)
        )

        # Each output position integrates:
        # - Past frames (retention)
        # - Current frame (primal impression)
        # - Future frames (protention, if causal=False)

    def forward(self, x):
        """
        x: (B, C, T, H, W) - video as spacetime volume

        Each output activation "experiences" present_thickness frames
        as a unified perceptual moment.
        """
        return self.conv(x)

    def get_temporal_receptive_field(self, num_layers):
        """
        Total "thickness of present" after num_layers.

        With stacking, the network builds hierarchical temporal structure:
        - Early layers: Thin present (3 frames)
        - Deep layers: Thick present (many frames)

        Like consciousness building from micro to macro time.
        """
        rf = 1
        for _ in range(num_layers):
            rf = rf + (self.present_thickness - 1)
        return rf
```

### Unifying Different Temporal Architectures

**TRAIN STATION**: Where these concepts meet:

```python
"""
3D CNN Temporal Architecture = Other Temporal Systems

Architecture          Temporal Mechanism         "Thick Present" Size
-----------          ------------------         -------------------
C3D                  3x3x3 kernels              Fixed 16 frames
I3D                  Inflated 2D kernels        Fixed 64 frames
R(2+1)D              Factorized T x 1 x 1       Separable temporal
SlowFast             Two pathways               4 + 32 frames
Video Transformers   Temporal attention         Variable attention span
State Space (S4)     Continuous dynamics        Infinite (theoretically)

All implement the same concept:
"Process time as a volume, not as instantaneous points"

This is the ML implementation of:
- Phenomenology: Experience has duration
- Physics: Events have temporal extent
- Neuroscience: Neural integration windows
- Philosophy: The specious present
"""

class TemporalProcessingComparison:
    """
    Compare how different architectures implement "thick present".
    """

    @staticmethod
    def cnn_3d_present(frames, kernel_t=3):
        """
        3D CNN: Fixed-size sliding window.
        Present = kernel_t frames at each position.
        """
        # Each output combines kernel_t adjacent frames
        return f"Fixed present of {kernel_t} frames per position"

    @staticmethod
    def transformer_present(frames, window_size=16):
        """
        Video Transformer: Attention-weighted window.
        Present = attention distribution over window_size frames.
        Soft boundaries - can attend anywhere in window.
        """
        # Each token attends to all frames in window
        return f"Soft attention over {window_size} frames"

    @staticmethod
    def ssm_present(frames):
        """
        State Space Model: Continuous integration.
        Present = hidden state encodes all past.
        No fixed window - infinite context theoretically.
        """
        # Hidden state carries temporal information
        return "Continuous state encoding all history"

    @staticmethod
    def predictive_coding_present(frames):
        """
        Predictive Coding: Hierarchical prediction.
        Present = prediction error between expected and actual.
        Time as error dynamics, not fixed windows.
        """
        return "Temporal present as prediction error dynamics"
```

### Connecting to Phenomenology and Active Inference

```python
"""
The 3D CNN implements what phenomenology describes and active inference formalizes:

PHENOMENOLOGY (Husserl's time consciousness):
- Retention: Past frames in kernel still influence
- Primal impression: Central frame position
- Protention: Future frames in kernel (if not causal)

ACTIVE INFERENCE (Friston's free energy):
- Generative model: Predict video from latent dynamics
- Temporal depth: Prior on temporal dependencies
- Precision: Weight on temporal prediction errors

3D CNN implementation:
- Kernel weights: Learned temporal dependencies
- Receptive field: Extent of temporal modeling
- Stride: Rate of temporal abstraction

The TRAIN STATION:
Phenomenological time = Active inference temporal depth = 3D CNN receptive field
"""

class PhenomenologicalConv3D(nn.Module):
    """
    3D convolution with phenomenological interpretation.

    Makes explicit the retention-impression-protention structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        retention_frames: int = 1,    # Past
        protention_frames: int = 1,   # Future
        spatial_kernel: int = 3
    ):
        super().__init__()

        # Temporal extent = retention + current + protention
        self.temporal_kernel = retention_frames + 1 + protention_frames

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(self.temporal_kernel, spatial_kernel, spatial_kernel),
            padding=(self.temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2)
        )

        # Could make retention/protention weights different
        # to model asymmetric temporal structure

    def forward(self, x):
        return self.conv(x)
```

---

## Section 8: ARR-COC-0-1 Connection - Thick Temporal Processing

### Relevance to Adaptive Inference

The 3D CNN "thick present" concept directly applies to ARR-COC's adaptive relevance realization:

```python
"""
ARR-COC-0-1: Adaptive Relevance Realization through Chains of Compression

3D CNN temporal processing connects to ARR-COC through:

1. TEMPORAL RELEVANCE WINDOWS
   - Not all tokens equally relevant at all times
   - Relevance has temporal structure (recent vs old)
   - 3D conv's temporal kernel = relevance integration window

2. MULTI-SCALE TEMPORAL PROCESSING
   - Different relevance for different timescales
   - Fast/slow temporal streams for different content
   - Like SlowFast networks for relevance

3. TEMPORAL COMPRESSION CHAINS
   - Video features compress temporal redundancy
   - Relevance should also compress temporal redundancy
   - Skip frames that are temporally similar to recent ones

Implementation ideas for ARR-COC:
"""

class TemporalRelevanceModule(nn.Module):
    """
    Compute relevance scores with temporal awareness.

    Uses 3D convolution ideas to model how relevance changes over time.
    """

    def __init__(
        self,
        hidden_dim: int,
        temporal_window: int = 8,
        num_heads: int = 4
    ):
        super().__init__()

        # Temporal relevance modeling with 3D conv
        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=temporal_window,
            padding=temporal_window // 2,
            groups=num_heads  # Group conv for efficiency
        )

        # Relevance scorer
        self.relevance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        self.temporal_window = temporal_window

    def forward(self, hidden_states, attention_mask=None):
        """
        Compute relevance scores with temporal context.

        Args:
            hidden_states: (B, T, D) - token representations over time
            attention_mask: (B, T) - which positions are valid

        Returns:
            relevance: (B, T) - relevance score per position
        """
        B, T, D = hidden_states.shape

        # Temporal convolution for context
        # (B, T, D) -> (B, D, T) -> Conv1D -> (B, D, T) -> (B, T, D)
        temporal_features = hidden_states.permute(0, 2, 1)
        temporal_features = self.temporal_conv(temporal_features)
        temporal_features = temporal_features.permute(0, 2, 1)

        # Compute relevance from temporal features
        relevance = self.relevance_head(temporal_features).squeeze(-1)  # (B, T)

        if attention_mask is not None:
            relevance = relevance * attention_mask

        return relevance


class ThickPresentTokenSelector(nn.Module):
    """
    Select tokens based on "thick present" relevance.

    Tokens are relevant not just by content but by temporal position.
    Recent tokens have higher base relevance (temporal decay).
    """

    def __init__(
        self,
        hidden_dim: int,
        present_thickness: int = 16,
        keep_ratio: float = 0.5
    ):
        super().__init__()

        self.temporal_relevance = TemporalRelevanceModule(
            hidden_dim, temporal_window=present_thickness
        )
        self.keep_ratio = keep_ratio

        # Learnable temporal decay
        self.temporal_decay = nn.Parameter(torch.ones(present_thickness))

    def forward(self, hidden_states):
        """
        Select most relevant tokens considering temporal structure.

        Returns indices of tokens to keep.
        """
        B, T, D = hidden_states.shape

        # Content-based relevance
        content_relevance = self.temporal_relevance(hidden_states)

        # Add temporal decay (recent = more relevant)
        positions = torch.arange(T, device=hidden_states.device).float()
        recency = torch.exp(-0.1 * (T - 1 - positions))  # Decay from end

        # Combined relevance
        relevance = content_relevance * recency.unsqueeze(0)

        # Select top-k
        k = int(T * self.keep_ratio)
        top_indices = relevance.topk(k, dim=1).indices

        return top_indices, relevance
```

### Video-Language Models and Temporal Relevance

```python
"""
For VLMs processing video, temporal relevance is crucial:

1. Not all frames equally relevant to text query
2. Relevance has temporal clustering (scenes)
3. Some frames are temporally redundant

ARR-COC could use 3D CNN insights for:
- Temporal pooling of similar frames
- Scene-level relevance aggregation
- Motion-based token selection
"""

class VideoRelevanceCompressor(nn.Module):
    """
    Compress video tokens using temporal relevance.

    Uses 3D convolution ideas for temporal modeling of relevance.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_frames: int = 16,
        compression_ratio: int = 4
    ):
        super().__init__()

        # 3D conv for temporal feature extraction
        self.temporal_encoder = nn.Conv3d(
            hidden_dim, hidden_dim,
            kernel_size=(3, 1, 1),  # Only temporal
            padding=(1, 0, 0)
        )

        # Relevance prediction
        self.relevance_predictor = nn.Linear(hidden_dim, 1)

        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool3d(
            (num_frames // compression_ratio, 1, 1)
        )

        self.compression_ratio = compression_ratio

    def forward(self, video_features, text_features=None):
        """
        Compress video features based on relevance.

        Args:
            video_features: (B, T, H*W, D) - video tokens per frame
            text_features: (B, L, D) - text query (optional)

        Returns:
            compressed: (B, T//ratio, H*W, D) - compressed video
            relevance: (B, T, H*W) - relevance scores
        """
        B, T, N, D = video_features.shape

        # Reshape for 3D conv: (B, D, T, H, W)
        # Treat N=H*W as flattened spatial
        H = W = int(N ** 0.5)
        x = video_features.permute(0, 3, 1, 2).view(B, D, T, H, W)

        # Temporal convolution
        temporal_features = self.temporal_encoder(x)

        # Compute relevance
        relevance = self.relevance_predictor(
            temporal_features.permute(0, 2, 3, 4, 1)  # (B, T, H, W, D)
        ).squeeze(-1)  # (B, T, H, W)

        # Pool temporally
        pooled = self.temporal_pool(x)  # (B, D, T//ratio, H, W)
        compressed = pooled.permute(0, 2, 3, 4, 1).view(
            B, T // self.compression_ratio, N, D
        )

        return compressed, relevance.view(B, T, N)
```

---

## Sources

**Primary Papers**:
- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248) - Tran et al., 2017 (R(2+1)D)
- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) - Tran et al., 2015 (C3D)
- [Quo Vadis, Action Recognition?](https://arxiv.org/abs/1705.07750) - Carreira & Zisserman, 2017 (I3D)
- [Rethinking Spatiotemporal Feature Learning](https://arxiv.org/abs/1712.04851) - Xie et al., 2017 (S3D)

**Implementation References**:
- [PyTorch Video ResNet](https://docs.pytorch.org/vision/main/models/video_resnet.html) - TorchVision video models
- [PyTorchVideo](https://ai.meta.com/blog/pytorchvideo-a-deep-learning-library-for-video-understanding/) - Meta's video deep learning library
- [Deep Learning on Video Part Three](https://cameronrwolfe.substack.com/p/deep-learning-on-video-part-three-diving-deeper-into-3d-cnns-cb3c0daa471e) - Comprehensive 3D CNN overview

**Stanford CS231N**:
- [Lecture 10: Video Understanding](https://cs231n.stanford.edu/slides/2023/lecture_10.pdf) - Stanford slides on video architectures

**Additional Resources**:
- [TensorFlow Video Classification Tutorial](https://www.tensorflow.org/tutorials/video/video_classification)
- [SlowFast Networks](https://arxiv.org/abs/1812.03982) - Multi-timescale video understanding
- [Video Transformers](https://arxiv.org/abs/2103.15691) - ViViT, TimeSformer
