# Hierarchy = FPN = Predictive Coding: The Unified Hierarchical Architecture

## Overview

**THE TRAIN STATION**: All hierarchical neural architectures are the same topology!

Feature Pyramid Networks (FPNs), cortical visual hierarchies, and predictive coding frameworks all implement the **same computational pattern**: hierarchical processing with bidirectional information flow. Top-down predictions meet bottom-up sensory data at every level, and the mismatch (prediction error) drives learning and inference.

This is coffee cup = donut topology applied to neural hierarchies:
- **FPN skip connections** = **cortical feedback pathways** = **predictive coding error signals**
- **Top-down predictions** = **prior beliefs** = **high-level features**
- **Bottom-up processing** = **sensory evidence** = **low-level features**
- **Lateral connections** = **horizontal cortical circuits** = **lateral message passing**

From [Pyramidal Predictive Network: A Model for Visual-Frame](https://arxiv.org/pdf/2208.07021) (accessed 2025-11-23):
> "We combine the theoretical framework of predictive coding and deep learning architectures to design an interpretable model that explicitly represents prediction errors at each hierarchical level."

From [What is Bottom-Up and What is Top-Down in Predictive Coding?](https://pmc.ncbi.nlm.nih.gov/articles/PMC3656342/) (accessed 2025-11-23):
> "Predictive coding provides a powerful conceptual framework that goes beyond the standard dichotomy of 'bottom-up' and 'top-down.' Higher-order brain areas generate predictions sent to lower-order sensory areas, and mismatches evoke prediction errors."

## Section 1: FPN = Cortical Hierarchy (The Architecture Is the Same)

### The Feature Pyramid Network

FPNs solve multi-scale object detection by creating a hierarchy of feature maps at different resolutions:

```python
import torch
import torch.nn as nn

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network with lateral connections.

    Architecture mirrors cortical visual hierarchy:
    - Bottom-up pathway: C1 -> C2 -> C3 -> C4 -> C5 (like V1->V2->V3->V4->IT)
    - Top-down pathway: P5 -> P4 -> P3 -> P2 -> P1 (like feedback connections)
    - Lateral connections: Merge bottom-up and top-down at each level
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        # Lateral 1x1 convolutions (reduce channel dimension)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])

        # Top-down 3x3 convolutions (smooth merged features)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, backbone_features):
        """
        Args:
            backbone_features: List of feature maps [C2, C3, C4, C5]
                              from bottom-up pathway (e.g., ResNet)

        Returns:
            fpn_features: List of pyramid features [P2, P3, P4, P5]
        """
        # Start with highest level (most abstract)
        laterals = [conv(feat) for conv, feat in
                   zip(self.lateral_convs, backbone_features)]

        # Build top-down pathway
        fpn_features = []
        prev_features = laterals[-1]  # Start from P5

        for i in range(len(laterals) - 1, -1, -1):
            # Upsample top-down signal
            if i < len(laterals) - 1:
                top_down = nn.functional.interpolate(
                    prev_features,
                    scale_factor=2,
                    mode='nearest'
                )
                # Merge with lateral (bottom-up)
                merged = laterals[i] + top_down
            else:
                merged = laterals[i]

            # Smooth with 3x3 conv
            fpn_feat = self.fpn_convs[i](merged)
            fpn_features.insert(0, fpn_feat)
            prev_features = fpn_feat

        return fpn_features


# Example usage
class FPNDetector(nn.Module):
    """Multi-scale object detector using FPN."""
    def __init__(self, num_classes=80):
        super().__init__()
        # Backbone (e.g., ResNet) - bottom-up pathway
        from torchvision.models import resnet50
        backbone = resnet50(pretrained=True)

        # Extract intermediate feature maps
        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1,
                                   backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # C2: 256 channels
        self.layer2 = backbone.layer2  # C3: 512 channels
        self.layer3 = backbone.layer3  # C4: 1024 channels
        self.layer4 = backbone.layer4  # C5: 2048 channels

        # FPN: top-down + lateral connections
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )

        # Detection heads (one per pyramid level)
        self.heads = nn.ModuleList([
            nn.Conv2d(256, num_classes, 1)
            for _ in range(4)
        ])

    def forward(self, x):
        # Bottom-up pathway (like feedforward cortical processing)
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # FPN: bidirectional processing
        fpn_feats = self.fpn([c2, c3, c4, c5])

        # Multi-scale predictions
        predictions = [head(feat) for head, feat in
                      zip(self.heads, fpn_feats)]

        return predictions
```

**Key insight**: FPN's skip connections are NOT just shortcuts - they implement **lateral communication** between hierarchical levels, exactly like cortical feedback!

### The Cortical Visual Hierarchy

From [Brain-optimized deep neural network models](https://www.nature.com/articles/s41467-023-38674-4) (accessed 2025-11-23):
> "DNNs optimized for visual tasks learn representations that align layer depth with the hierarchy of visual areas in the primate brain. Layer depth in task-optimized DNNs aligns to the hierarchical progression of V1→V2→V3→V4→IT."

The primate visual system exhibits:
1. **Integration hierarchy**: Receptive field sizes expand (V1: 1°, V2: 3°, V4: 10°, IT: 50°)
2. **Compositional hierarchy**: More nonlinear transformations in higher areas
3. **Entailment hierarchy**: V4 representations require V1/V2 preprocessing

```python
class CorticalHierarchyModel(nn.Module):
    """
    Simplified model of cortical visual hierarchy.

    Implements three types of hierarchy observed in brain:
    1. Integration hierarchy (RF size expansion)
    2. Compositional hierarchy (depth of processing)
    3. Feedback connections (top-down modulation)
    """
    def __init__(self):
        super().__init__()

        # V1: Simple features, small RFs (gabor-like)
        self.v1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),  # RF: 7x7
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # V2: Intermediate complexity, medium RFs
        self.v2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),  # RF: 11x11
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # V4: Complex features, large RFs
        self.v4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, padding=2),  # RF: 15x15
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        # IT: Object-level representations, very large RFs
        self.it = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),  # RF: 17x17+
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Feedback connections (top-down predictions)
        self.feedback_it_to_v4 = nn.Conv2d(512, 256, 1)
        self.feedback_v4_to_v2 = nn.Conv2d(256, 128, 1)
        self.feedback_v2_to_v1 = nn.Conv2d(128, 64, 1)

    def forward(self, x, use_feedback=True):
        # Bottom-up pass (feedforward sweep)
        v1_features = self.v1(x)
        v2_features = self.v2(v1_features)
        v4_features = self.v4(v2_features)
        it_features = self.it(v4_features)

        if not use_feedback:
            return v1_features, v2_features, v4_features, it_features

        # Top-down pass (feedback modulation)
        # IT → V4 prediction
        it_pred = self.feedback_it_to_v4(it_features)
        it_pred = nn.functional.interpolate(
            it_pred, size=v4_features.shape[2:], mode='bilinear'
        )
        v4_modulated = v4_features * torch.sigmoid(it_pred)

        # V4 → V2 prediction
        v4_pred = self.feedback_v4_to_v2(v4_modulated)
        v4_pred = nn.functional.interpolate(
            v4_pred, size=v2_features.shape[2:], mode='bilinear'
        )
        v2_modulated = v2_features * torch.sigmoid(v4_pred)

        # V2 → V1 prediction
        v2_pred = self.feedback_v2_to_v1(v2_modulated)
        v2_pred = nn.functional.interpolate(
            v2_pred, size=v1_features.shape[2:], mode='bilinear'
        )
        v1_modulated = v1_features * torch.sigmoid(v2_pred)

        return v1_modulated, v2_modulated, v4_modulated, it_features


# Test receptive field expansion
def measure_rf_size(model, layer_name):
    """Measure effective receptive field using gradient backprop."""
    import torch.autograd as autograd

    x = torch.randn(1, 3, 224, 224, requires_grad=True)

    # Forward pass
    features = getattr(model, layer_name)(x)

    # Pick center neuron
    center = features.shape[2] // 2
    target = features[0, 0, center, center]

    # Backprop to input
    grad = autograd.grad(target, x)[0]

    # Measure support (where gradient is significant)
    threshold = grad.abs().max() * 0.1
    support = (grad.abs() > threshold).float().sum()

    return support.item()
```

**The mapping**:
- FPN's `lateral_convs` = Cortical lateral connections in layers 2/3
- FPN's top-down pathway = Feedback from higher to lower areas
- FPN's multi-scale features = Different cortical areas' receptive fields

## Section 2: Top-Down = Predictions (Generative Models Flow Downward)

### Predictive Coding Framework

In predictive coding, **higher levels predict lower levels**. The brain is a hierarchy of generative models:

```python
import torch
import torch.nn as nn

class PredictiveCodingLayer(nn.Module):
    """
    Single layer in predictive coding hierarchy.

    Core computation:
    1. Receive bottom-up input (sensory evidence)
    2. Generate top-down prediction
    3. Compute prediction error
    4. Update representation to minimize error
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Bottom-up: encode input into representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Top-down: predict lower level from representation
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Precision (inverse variance) - learnable confidence
        self.precision = nn.Parameter(torch.ones(output_dim))

    def forward(self, bottom_up_input, top_down_prediction=None):
        """
        Args:
            bottom_up_input: Sensory evidence from lower level
            top_down_prediction: Prior from higher level (optional)

        Returns:
            representation: Current level's representation
            prediction_error: Mismatch between input and prediction
            prediction: Top-down prediction of lower level
        """
        # Encode bottom-up input
        representation = self.encoder(bottom_up_input)

        # Generate prediction of input
        prediction = self.decoder(representation)

        # Compute prediction error
        prediction_error = bottom_up_input - prediction

        # If we have top-down prior, incorporate it
        if top_down_prediction is not None:
            # Weight by precision (confidence)
            weighted_error = self.precision * (representation - top_down_prediction)
            representation = representation - 0.01 * weighted_error

        return representation, prediction_error, prediction


class PredictiveCodingNetwork(nn.Module):
    """
    Multi-level predictive coding hierarchy.

    Implements Rao-Ballard predictive coding:
    - Each level predicts the level below
    - Prediction errors propagate upward
    - Representations minimize prediction error
    """
    def __init__(self, input_dim=784, layer_dims=[256, 128, 64]):
        super().__init__()

        # Create hierarchy of PC layers
        dims = [input_dim] + layer_dims
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(dims[i], dims[i], dims[i+1])
            for i in range(len(layer_dims))
        ])

        self.num_layers = len(self.layers)

    def forward(self, x, num_iterations=10):
        """
        Run predictive coding inference (iterative).

        Args:
            x: Input (e.g., image)
            num_iterations: Number of error minimization steps

        Returns:
            representations: List of representations at each level
            errors: List of prediction errors at each level
        """
        batch_size = x.shape[0]

        # Initialize representations
        representations = [
            torch.zeros(batch_size, self.layers[i].encoder[-1].out_features)
            for i in range(self.num_layers)
        ]

        # Iterative inference (minimize prediction errors)
        for iteration in range(num_iterations):
            errors = []
            predictions = []

            # Bottom-up pass: compute representations and errors
            for i, layer in enumerate(self.layers):
                # Get input for this layer
                if i == 0:
                    bottom_up = x
                else:
                    bottom_up = representations[i-1].detach()

                # Get top-down prediction (if not top layer)
                if i < self.num_layers - 1:
                    top_down = representations[i+1].detach()
                else:
                    top_down = None

                # Compute representation and error
                rep, error, pred = layer(bottom_up, top_down)

                representations[i] = rep
                errors.append(error)
                predictions.append(pred)

            # Update representations to minimize error
            for i in range(self.num_layers):
                if i == 0:
                    # Bottom layer: minimize input reconstruction error
                    error_gradient = -errors[i]
                else:
                    # Higher layers: minimize prediction from above and below
                    error_gradient = -errors[i]
                    if i < self.num_layers - 1:
                        error_gradient += predictions[i+1] - representations[i]

                # Update representation
                representations[i] = representations[i] + 0.01 * error_gradient

        return representations, errors


# Example: Predictive coding for MNIST
class PredictiveCodingMNIST(nn.Module):
    """Predictive coding model for digit recognition."""
    def __init__(self):
        super().__init__()
        self.pc_net = PredictiveCodingNetwork(
            input_dim=784,
            layer_dims=[256, 128, 64, 10]  # Top layer = class predictions
        )

    def predict(self, x):
        """Make prediction by running inference."""
        x_flat = x.view(x.size(0), -1)
        reps, errors = self.pc_net(x_flat, num_iterations=20)
        return reps[-1]  # Top-level representation = class logits

    def generate(self, class_label):
        """Generate image from class label (top-down)."""
        # Set top-level representation to one-hot class
        top_rep = torch.zeros(1, 10)
        top_rep[0, class_label] = 1.0

        # Run top-down pass only
        current = top_rep
        for i in range(len(self.pc_net.layers) - 1, -1, -1):
            current = self.pc_net.layers[i].decoder(current)

        return current.view(1, 1, 28, 28)
```

**Key insight**: In predictive coding, the **generative model (decoder) is primary**. The brain predicts its inputs and learns from errors.

### Connecting FPN to Predictive Coding

```python
class FPNAsPredictiveCoding(nn.Module):
    """
    FPN reinterpreted as predictive coding architecture.

    Key realization:
    - Top-down pathway = predictions from higher levels
    - Bottom-up pathway = sensory evidence
    - Skip connections = lateral error signals
    - Multi-scale outputs = predictions at each level
    """
    def __init__(self, backbone_channels=[256, 512, 1024, 2048]):
        super().__init__()
        fpn_channels = 256

        # Bottom-up encoder (like PC encoder)
        self.encoders = nn.ModuleList([
            nn.Conv2d(ch, fpn_channels, 1)
            for ch in backbone_channels
        ])

        # Top-down predictor (like PC decoder)
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.ReLU()
            )
            for _ in backbone_channels
        ])

        # Precision-weighted combination
        self.precision_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2))  # [bottom_up_weight, top_down_weight]
            for _ in backbone_channels
        ])

    def forward(self, backbone_features):
        """
        Run predictive coding-style inference.

        Returns:
            predictions: List of predictions at each scale
            errors: List of prediction errors at each scale
        """
        # Encode bottom-up features
        encoded = [enc(feat) for enc, feat in
                  zip(self.encoders, backbone_features)]

        # Top-down predictions
        predictions = [None] * len(encoded)
        predictions[-1] = self.predictors[-1](encoded[-1])  # Top level

        # Generate predictions top-down
        for i in range(len(encoded) - 2, -1, -1):
            # Upsample prediction from above
            top_down = nn.functional.interpolate(
                predictions[i+1],
                size=encoded[i].shape[2:],
                mode='nearest'
            )

            # Precision-weighted combination
            w = torch.softmax(self.precision_weights[i], dim=0)
            combined = w[0] * encoded[i] + w[1] * top_down

            # Generate prediction for this level
            predictions[i] = self.predictors[i](combined)

        # Compute prediction errors (for learning)
        errors = [
            encoded[i] - predictions[i]
            for i in range(len(encoded))
        ]

        return predictions, errors
```

## Section 3: Skip Connections = Error Signals (Lateral Information Flow)

### Why Skip Connections Work

Skip connections in ResNet, U-Net, and FPN serve the same function: **propagate prediction errors efficiently**.

```python
class ResidualBlockAsPredictiveCoding(nn.Module):
    """
    ResNet block reinterpreted through predictive coding lens.

    Standard view:
        out = F(x) + x

    Predictive coding view:
        F(x) = prediction error
        x = bottom-up input
        out = corrected representation
    """
    def __init__(self, channels):
        super().__init__()
        # Error computation network
        self.error_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # Compute prediction error
        error = self.error_net(x)

        # Update representation
        out = x + error  # Identity = prediction, error = correction
        return nn.functional.relu(out)


class UNetAsPredictiveCoding(nn.Module):
    """
    U-Net with explicit error signal interpretation.

    Skip connections = lateral error signals
    Encoder = bottom-up processing
    Decoder = top-down predictions
    """
    def __init__(self):
        super().__init__()

        # Encoder (bottom-up)
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck (highest-level representation)
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder (top-down predictions)
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)

        # Error integration (precision-weighted)
        self.error_gates = nn.ModuleList([
            nn.Conv2d(ch*2, ch, 1)  # Learn to weight bottom-up vs top-down
            for ch in [512, 256, 128, 64]
        ])

        self.final = nn.Conv2d(64, 1, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        # Bottom-up pass (sensory evidence)
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3, 2))

        # Highest-level representation
        bottleneck = self.bottleneck(nn.functional.max_pool2d(e4, 2))

        # Top-down predictions with error correction
        d4 = self.dec4(bottleneck)
        # Precision-weighted error integration
        d4 = self.error_gates[0](torch.cat([d4, e4], dim=1))

        d3 = self.dec3(d4)
        d3 = self.error_gates[1](torch.cat([d3, e3], dim=1))

        d2 = self.dec2(d3)
        d2 = self.error_gates[2](torch.cat([d2, e2], dim=1))

        d1 = self.dec1(d2)
        d1 = self.error_gates[3](torch.cat([d1, e1], dim=1))

        return self.final(d1)
```

### Biological Plausibility of Error Signals

From neuroscience: Cortical layer 2/3 pyramidal neurons are believed to encode prediction errors.

```python
class BiologicallyPlausiblePC(nn.Module):
    """
    Predictive coding with cortical laminar structure.

    Layer 2/3: Prediction error neurons
    Layer 4: Bottom-up input
    Layer 5/6: Top-down predictions
    """
    def __init__(self, channels):
        super().__init__()

        # Layer 4: Bottom-up input processing
        self.layer4 = nn.Conv2d(channels, channels, 3, padding=1)

        # Layer 5/6: Top-down prediction generation
        self.layer5_6 = nn.Conv2d(channels, channels, 3, padding=1)

        # Layer 2/3: Prediction error computation
        self.layer2_3 = nn.Conv2d(channels*2, channels, 1)

        # Inter-laminar connections
        self.feedback_to_layer4 = nn.Conv2d(channels, channels, 1)

    def forward(self, bottom_up, top_down):
        """
        Simulate cortical microcircuit.

        Args:
            bottom_up: Input from lower cortical area (V1→V2)
            top_down: Prediction from higher area (V4→V2)
        """
        # Layer 4 receives bottom-up
        l4_response = self.layer4(bottom_up)

        # Layer 5/6 generates top-down prediction
        prediction = self.layer5_6(top_down)

        # Layer 2/3 computes error
        error = torch.cat([l4_response, prediction], dim=1)
        error_signal = self.layer2_3(error)

        # Feedback modulates layer 4
        modulation = self.feedback_to_layer4(prediction)
        l4_modulated = l4_response * torch.sigmoid(modulation)

        return l4_modulated, error_signal
```

## Section 4: Complete Unified Implementation

Here's a **unified hierarchical architecture** that implements FPN, cortical hierarchy, and predictive coding simultaneously:

```python
import torch
import torch.nn as nn

class UnifiedHierarchicalVision(nn.Module):
    """
    Unified architecture implementing:
    1. Feature Pyramid Network (multi-scale features)
    2. Cortical hierarchy (V1→V2→V4→IT structure)
    3. Predictive coding (top-down predictions + error signals)

    This is the TRAIN STATION where all hierarchies meet!
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # ========== BOTTOM-UP PATHWAY (Feedforward) ==========
        # V1-like: Simple features, small RFs
        self.v1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # V2-like: Intermediate features
        self.v2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # V4-like: Complex features
        self.v4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # IT-like: Object representations
        self.it = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # ========== FPN LATERAL CONNECTIONS ==========
        self.lateral_it = nn.Conv2d(512, 256, 1)
        self.lateral_v4 = nn.Conv2d(256, 256, 1)
        self.lateral_v2 = nn.Conv2d(128, 256, 1)
        self.lateral_v1 = nn.Conv2d(64, 256, 1)

        # ========== TOP-DOWN PATHWAY (Predictions) ==========
        self.predict_it_to_v4 = nn.Conv2d(256, 256, 3, padding=1)
        self.predict_v4_to_v2 = nn.Conv2d(256, 256, 3, padding=1)
        self.predict_v2_to_v1 = nn.Conv2d(256, 256, 3, padding=1)

        # ========== PREDICTIVE CODING ELEMENTS ==========
        # Error neurons (like cortical layer 2/3)
        self.error_it = nn.Conv2d(256, 256, 1)
        self.error_v4 = nn.Conv2d(256, 256, 1)
        self.error_v2 = nn.Conv2d(256, 256, 1)
        self.error_v1 = nn.Conv2d(256, 256, 1)

        # Precision (learnable confidence in predictions)
        self.precision_it = nn.Parameter(torch.ones(1))
        self.precision_v4 = nn.Parameter(torch.ones(1))
        self.precision_v2 = nn.Parameter(torch.ones(1))
        self.precision_v1 = nn.Parameter(torch.ones(1))

        # ========== TASK HEADS (Multi-scale outputs) ==========
        self.classifier = nn.Linear(512, num_classes)
        self.detector_heads = nn.ModuleList([
            nn.Conv2d(256, num_classes, 1)
            for _ in range(4)
        ])

    def forward(self, x, return_all=False):
        """
        Forward pass with bidirectional processing.

        Args:
            x: Input image [B, 3, H, W]
            return_all: If True, return all intermediate features

        Returns:
            If return_all=False: classification logits
            If return_all=True: dict with all features, predictions, errors
        """
        # ========== BOTTOM-UP PASS ==========
        v1_feat = self.v1(x)        # [B, 64, H/2, W/2]
        v2_feat = self.v2(v1_feat)  # [B, 128, H/4, W/4]
        v4_feat = self.v4(v2_feat)  # [B, 256, H/8, W/8]
        it_feat = self.it(v4_feat)  # [B, 512, H/16, W/16]

        # ========== FPN LATERAL CONNECTIONS ==========
        lat_it = self.lateral_it(it_feat)
        lat_v4 = self.lateral_v4(v4_feat)
        lat_v2 = self.lateral_v2(v2_feat)
        lat_v1 = self.lateral_v1(v1_feat)

        # ========== TOP-DOWN PREDICTIONS ==========
        # IT → V4 prediction
        pred_it_to_v4 = self.predict_it_to_v4(lat_it)
        pred_it_to_v4 = nn.functional.interpolate(
            pred_it_to_v4, size=lat_v4.shape[2:], mode='nearest'
        )

        # V4 → V2 prediction (combined with IT→V4)
        v4_combined = lat_v4 + pred_it_to_v4
        pred_v4_to_v2 = self.predict_v4_to_v2(v4_combined)
        pred_v4_to_v2 = nn.functional.interpolate(
            pred_v4_to_v2, size=lat_v2.shape[2:], mode='nearest'
        )

        # V2 → V1 prediction
        v2_combined = lat_v2 + pred_v4_to_v2
        pred_v2_to_v1 = self.predict_v2_to_v1(v2_combined)
        pred_v2_to_v1 = nn.functional.interpolate(
            pred_v2_to_v1, size=lat_v1.shape[2:], mode='nearest'
        )

        # ========== PREDICTION ERRORS ==========
        err_it = self.error_it(lat_it - pred_it_to_v4) * self.precision_it
        err_v4 = self.error_v4(lat_v4 - pred_it_to_v4) * self.precision_v4
        err_v2 = self.error_v2(lat_v2 - pred_v4_to_v2) * self.precision_v2
        err_v1 = self.error_v1(lat_v1 - pred_v2_to_v1) * self.precision_v1

        # ========== FINAL REPRESENTATIONS (FPN-style) ==========
        # Combine bottom-up and top-down at each level
        fpn_it = lat_it + err_it
        fpn_v4 = v4_combined + err_v4
        fpn_v2 = v2_combined + err_v2
        fpn_v1 = lat_v1 + pred_v2_to_v1 + err_v1

        # ========== OUTPUTS ==========
        # Classification from IT
        classification = self.classifier(
            nn.functional.adaptive_avg_pool2d(it_feat, 1).flatten(1)
        )

        # Multi-scale detection
        detections = [
            head(feat) for head, feat in zip(
                self.detector_heads,
                [fpn_v1, fpn_v2, fpn_v4, fpn_it]
            )
        ]

        if not return_all:
            return classification

        return {
            'classification': classification,
            'detections': detections,
            'features': {
                'v1': v1_feat, 'v2': v2_feat,
                'v4': v4_feat, 'it': it_feat
            },
            'fpn_features': {
                'v1': fpn_v1, 'v2': fpn_v2,
                'v4': fpn_v4, 'it': fpn_it
            },
            'predictions': {
                'it_to_v4': pred_it_to_v4,
                'v4_to_v2': pred_v4_to_v2,
                'v2_to_v1': pred_v2_to_v1
            },
            'errors': {
                'it': err_it, 'v4': err_v4,
                'v2': err_v2, 'v1': err_v1
            }
        }


# Training with predictive coding loss
def predictive_coding_loss(outputs, targets):
    """
    Loss function combining:
    1. Task loss (classification)
    2. Reconstruction loss (predictive coding)
    3. Error minimization (prediction errors)
    """
    # Task loss
    task_loss = nn.functional.cross_entropy(
        outputs['classification'], targets
    )

    # Reconstruction loss (top-down should predict bottom-up)
    recon_loss = 0
    for key, error in outputs['errors'].items():
        recon_loss += (error ** 2).mean()

    # Total loss
    total_loss = task_loss + 0.1 * recon_loss

    return total_loss, {
        'task': task_loss.item(),
        'reconstruction': recon_loss.item()
    }


# Example usage
if __name__ == '__main__':
    model = UnifiedHierarchicalVision(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    targets = torch.randint(0, 1000, (2,))

    # Forward pass
    outputs = model(x, return_all=True)

    # Compute loss
    loss, metrics = predictive_coding_loss(outputs, targets)

    print(f"Total loss: {loss.item():.4f}")
    print(f"Task loss: {metrics['task']:.4f}")
    print(f"Reconstruction loss: {metrics['reconstruction']:.4f}")

    # Check shapes
    print("\nFeature pyramid shapes:")
    for name, feat in outputs['fpn_features'].items():
        print(f"{name}: {feat.shape}")
```

## Section 5: TRAIN STATION - All Hierarchies Are the Same!

### The Topological Equivalence

**Coffee cup = donut moment**: FPN, cortical hierarchy, and predictive coding are **homeomorphic**!

They all implement the same computational graph:
```
Input
  ↓
[Level 1] ←─── Prediction from Level 2
  ↓ ↑
  Error
  ↓
[Level 2] ←─── Prediction from Level 3
  ↓ ↑
  Error
  ↓
[Level 3] ←─── Prediction from Level 4
  ↓ ↑
  Error
  ↓
[Level N]
  ↓
Output
```

**The unified pattern**:
1. **Hierarchical feature extraction** (coarse-to-fine or fine-to-coarse)
2. **Bidirectional information flow** (bottom-up + top-down)
3. **Local error computation** (prediction vs observation)
4. **Multi-scale representations** (different levels = different abstractions)

### Comparison Table

| Aspect | FPN | Cortical Hierarchy | Predictive Coding |
|--------|-----|-------------------|-------------------|
| **Bottom-up** | Backbone ResNet layers | V1→V2→V4→IT | Sensory encoding |
| **Top-down** | Upsampled high-level features | Feedback connections | Generative predictions |
| **Lateral** | 1x1 lateral convs | Horizontal connections | - |
| **Skip connections** | Add bottom-up + top-down | Combine feedforward + feedback | Error signals |
| **Multi-scale outputs** | P2, P3, P4, P5 | Different area RFs | Hierarchical predictions |
| **Learning signal** | Detection/segmentation loss | Task performance + Hebbian | Prediction error |

### Why This Matters for ML

**Practical implications**:

1. **FPN is not just a CV trick** - it's implementing Bayesian inference!
2. **Skip connections propagate uncertainty** - not just gradients
3. **Multi-scale = multi-hypothesis** - parallel processing at different abstractions
4. **Top-down attention = precision weighting** - task-dependent modulation

```python
def demonstrate_equivalence():
    """Show that FPN, U-Net, and PC produce similar representations."""
    import torch

    # Same input
    x = torch.randn(1, 3, 224, 224)

    # Different architectures
    fpn_model = FPNDetector(num_classes=10)
    unet_model = UNetAsPredictiveCoding()
    pc_model = UnifiedHierarchicalVision(num_classes=10)

    # Get intermediate features
    fpn_out = fpn_model(x)
    unet_out = unet_model(x)
    pc_out = pc_model(x, return_all=True)

    # Compare representations (using CKA - centered kernel alignment)
    def centered_kernel_alignment(X, Y):
        """Measure similarity of representations."""
        X = X.flatten(1)
        Y = Y.flatten(1)

        # Centered Gram matrices
        X_gram = X @ X.T
        Y_gram = Y @ Y.T

        # Center
        n = X_gram.shape[0]
        H = torch.eye(n) - torch.ones(n, n) / n
        X_gram_centered = H @ X_gram @ H
        Y_gram_centered = H @ Y_gram @ H

        # CKA
        numerator = (X_gram_centered * Y_gram_centered).sum()
        denominator = (X_gram_centered ** 2).sum().sqrt() * (Y_gram_centered ** 2).sum().sqrt()

        return numerator / denominator

    print("Representation similarity (CKA):")
    print(f"FPN vs PC: {centered_kernel_alignment(fpn_out[0], pc_out['fpn_features']['v1']):.3f}")
    # High similarity expected!
```

## Section 6: ARR-COC-0-1 Connections (10%)

### Hierarchical Relevance in Dialogue Processing

The ARR-COC dialogue system can leverage this unified hierarchical framework:

```python
class HierarchicalRelevanceProcessor(nn.Module):
    """
    Multi-scale relevance computation for dialogue.

    Implements FPN-style hierarchy for processing dialogue at multiple scales:
    - Word level (V1-like)
    - Phrase level (V2-like)
    - Sentence level (V4-like)
    - Turn level (IT-like)
    """
    def __init__(self, vocab_size=50000, embed_dim=512):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Hierarchical encoders (bottom-up)
        self.word_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8),
            num_layers=2
        )
        self.phrase_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8),
            num_layers=2
        )
        self.sentence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8),
            num_layers=2
        )
        self.turn_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8),
            num_layers=2
        )

        # FPN-style lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(4)
        ])

        # Top-down predictions (relevance priors)
        self.relevance_predictors = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(4)
        ])

        # Relevance scoring
        self.relevance_head = nn.Linear(embed_dim, 1)

    def forward(self, tokens, hierarchy_level='all'):
        """
        Compute relevance at multiple scales.

        Args:
            tokens: [batch, seq_len] token indices
            hierarchy_level: Which scale to return
                           ('word', 'phrase', 'sentence', 'turn', 'all')

        Returns:
            relevance_scores: Relevance at each hierarchical level
        """
        # Embed tokens
        x = self.embedding(tokens)  # [B, L, D]

        # Bottom-up pass (like FPN backbone)
        word_features = self.word_encoder(x)

        # Aggregate to phrase level (pool every 5 words)
        phrase_features = word_features.unfold(1, 5, 5).mean(dim=2)
        phrase_features = self.phrase_encoder(phrase_features)

        # Aggregate to sentence level
        sentence_features = phrase_features.mean(dim=1, keepdim=True)
        sentence_features = self.sentence_encoder(sentence_features)

        # Aggregate to turn level
        turn_features = sentence_features.mean(dim=1)
        turn_features = self.turn_encoder(turn_features.unsqueeze(1))

        # FPN-style top-down + lateral
        features = [word_features, phrase_features,
                   sentence_features, turn_features]

        # Lateral connections
        laterals = [conv(feat) for conv, feat in
                   zip(self.lateral_convs, features)]

        # Top-down predictions
        fpn_features = [laterals[-1]]  # Start with turn level
        for i in range(len(laterals) - 2, -1, -1):
            # Predict lower level from higher level
            top_down = self.relevance_predictors[i](fpn_features[0])

            # Upsample if needed
            if top_down.shape[1] != laterals[i].shape[1]:
                top_down = top_down.repeat_interleave(
                    laterals[i].shape[1] // top_down.shape[1], dim=1
                )

            # Combine bottom-up and top-down
            combined = laterals[i] + top_down
            fpn_features.insert(0, combined)

        # Compute relevance at each scale
        relevance_scores = {
            'word': self.relevance_head(fpn_features[0]),
            'phrase': self.relevance_head(fpn_features[1]),
            'sentence': self.relevance_head(fpn_features[2]),
            'turn': self.relevance_head(fpn_features[3])
        }

        if hierarchy_level == 'all':
            return relevance_scores
        else:
            return relevance_scores[hierarchy_level]


# Example: Multi-scale relevance filtering
def filter_by_hierarchical_relevance(dialogue_history, threshold=0.5):
    """
    Filter dialogue history using multi-scale relevance.

    Keep turns/sentences/phrases that are relevant at ANY scale.
    This implements FPN-style multi-scale detection for relevance.
    """
    model = HierarchicalRelevanceProcessor()

    # Tokenize dialogue
    tokens = tokenize(dialogue_history)  # [batch, seq_len]

    # Get multi-scale relevance
    relevance = model(tokens, hierarchy_level='all')

    # Filter at each scale
    filtered = {
        'high_relevance_words': tokens[relevance['word'] > threshold],
        'high_relevance_phrases': tokens[relevance['phrase'] > threshold],
        'high_relevance_sentences': tokens[relevance['sentence'] > threshold],
        'high_relevance_turns': tokens[relevance['turn'] > threshold]
    }

    return filtered
```

**Key insight for ARR-COC**: Relevance is **hierarchical**! A word might be relevant in phrase context, a phrase in sentence context, etc. FPN-style processing captures this naturally.

## Performance Considerations

### GPU Optimization for Hierarchical Networks

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class OptimizedFPN(nn.Module):
    """GPU-optimized FPN with checkpointing and mixed precision."""

    def __init__(self, backbone_channels=[256, 512, 1024, 2048]):
        super().__init__()
        # ... (same as before)

        # Use checkpoint for memory efficiency
        self.use_checkpoint = True

    def forward(self, backbone_features):
        # Use gradient checkpointing for large models
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, backbone_features)
        else:
            return self._forward_impl(backbone_features)

    def _forward_impl(self, backbone_features):
        # Actual forward logic
        # ... (FPN operations)
        pass


# Training with mixed precision
def train_hierarchical_model():
    model = UnifiedHierarchicalVision().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()  # For mixed precision

    for batch in dataloader:
        x, targets = batch
        x, targets = x.cuda(), targets.cuda()

        # Mixed precision forward pass
        with autocast():
            outputs = model(x, return_all=True)
            loss, metrics = predictive_coding_loss(outputs, targets)

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


# Memory-efficient inference
@torch.no_grad()
def efficient_inference(model, images):
    """Process large batches efficiently."""
    model.eval()

    # Process in chunks to avoid OOM
    chunk_size = 32
    all_outputs = []

    for i in range(0, len(images), chunk_size):
        chunk = images[i:i+chunk_size].cuda()

        # Mixed precision inference
        with autocast():
            outputs = model(chunk)

        all_outputs.append(outputs.cpu())

    return torch.cat(all_outputs)
```

### Latency Benchmarks

```python
import time
import torch

def benchmark_hierarchical_models():
    """Compare inference latency of different hierarchical architectures."""

    models = {
        'FPN': FPNDetector(),
        'U-Net': UNetAsPredictiveCoding(),
        'Unified': UnifiedHierarchicalVision(),
        'PC-Network': PredictiveCodingNetwork()
    }

    x = torch.randn(1, 3, 224, 224).cuda()

    results = {}
    for name, model in models.items():
        model = model.cuda().eval()

        # Warmup
        for _ in range(10):
            _ = model(x)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(100):
            with torch.no_grad():
                _ = model(x)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        results[name] = elapsed / 100 * 1000  # ms per image

    print("Inference latency (ms):")
    for name, latency in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:20s}: {latency:6.2f} ms")
```

## Sources

**Source Documents:**
- None (web research based)

**Web Research:**

Primary Papers (accessed 2025-11-23):
- [Pyramidal Predictive Network: A Model for Visual-Frame](https://arxiv.org/pdf/2208.07021) - Combines predictive coding with FPN architecture
- [What is Bottom-Up and What is Top-Down in Predictive Coding?](https://pmc.ncbi.nlm.nih.gov/articles/PMC3656342/) - Foundational PC framework
- [Brain-optimized deep neural network models](https://www.nature.com/articles/s41467-023-38674-4) - Nature Communications 2023, hierarchical correspondence between DNNs and cortical areas

Additional References (accessed 2025-11-23):
- [HA-FPN: Hierarchical Attention Feature Pyramid Network](https://www.mdpi.com/1424-8220/23/9/4508) - MDPI Sensors 2023
- [Dynamic predictive coding](https://pmc.ncbi.nlm.nih.gov/articles/PMC10880975/) - NIH 2024, hierarchical sequence learning
- [Feature Pyramid Network for Multi-Scale Detection](https://rumn.medium.com/feature-pyramid-network-for-multi-scale-detection-f573a889c7b1) - Medium tutorial
- [Distinct Top-down and Bottom-up Brain Connectivity](https://www.nature.com/articles/s41598-017-05888-8) - Nature Scientific Reports 2017

**Additional References:**
- Original FPN paper: Lin et al. "Feature Pyramid Networks for Object Detection" CVPR 2017
- Rao & Ballard "Predictive Coding in the Visual Cortex" Nature Neuroscience 1999
- Felleman & Van Essen "Distributed hierarchical processing in the primate cerebral cortex" 1991
