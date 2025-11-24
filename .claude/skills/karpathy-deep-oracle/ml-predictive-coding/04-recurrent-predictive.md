# Recurrent Predictive Networks: Temporal Prediction Through Predictive Coding

## Overview

Recurrent predictive networks combine the temporal modeling power of recurrent neural networks with the principles of predictive coding to achieve state-of-the-art video prediction and temporal sequence forecasting. These architectures learn to predict future frames by minimizing prediction errors at multiple levels of a hierarchical representation.

**Key Insight**: The brain doesn't just process what it sees - it constantly predicts what it will see next. Recurrent predictive networks formalize this principle in deep learning.

---

## 1. PredNet Architecture: Predictive Coding Meets Deep Learning

### 1.1 Core Architecture Principles

PredNet (Lotter et al., 2016) implements predictive coding principles in a deep convolutional recurrent network:

From [PredNet by coxlab](https://coxlab.github.io/prednet/) and [Deep Predictive Coding Networks for Video Prediction](https://arxiv.org/abs/1605.08104):

**Architecture Components:**

```
Layer l contains four types of units:
1. Representation units (R_l) - ConvLSTM cells
2. Prediction units (A-hat_l) - Generate predictions
3. Input/Target units (A_l) - What we're trying to predict
4. Error units (E_l) - Compute prediction errors
```

**Information Flow:**

```
Bottom-up: E_l -> R_{l+1}  (errors drive higher representations)
Top-down: R_l -> A-hat_l   (representations generate predictions)
Lateral: R_l -> R_l        (recurrent temporal dynamics)
```

### 1.2 PyTorch PredNet Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatial-temporal processing."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Combined gates for efficiency
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)

        # Split into individual gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Cell state update
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class PredNetLayer(nn.Module):
    """Single layer of PredNet with prediction and error computation."""

    def __init__(self, input_channels, repr_channels, error_channels):
        super().__init__()

        # Representation unit (ConvLSTM)
        self.repr_lstm = ConvLSTMCell(
            error_channels + repr_channels,  # Bottom-up error + top-down repr
            repr_channels
        )

        # Prediction unit
        self.prediction = nn.Sequential(
            nn.Conv2d(repr_channels, repr_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(repr_channels, input_channels, 3, padding=1)
        )

        # Error computation (ReLU split into pos/neg)
        self.error_channels = error_channels

    def forward(self, a_target, e_below, r_above, state):
        """
        Args:
            a_target: Target to predict (input at this level)
            e_below: Error from layer below (upsampled)
            r_above: Representation from layer above (downsampled)
            state: (h, c) tuple for ConvLSTM
        """
        # Combine inputs to representation unit
        if r_above is not None:
            r_input = torch.cat([e_below, r_above], dim=1)
        else:
            r_input = e_below

        # Update representation
        r, new_state = self.repr_lstm(r_input, state)

        # Generate prediction
        a_hat = self.prediction(r)

        # Compute error (positive and negative separately)
        error = a_target - a_hat
        e_pos = F.relu(error)
        e_neg = F.relu(-error)
        e = torch.cat([e_pos, e_neg], dim=1)

        return r, e, a_hat, new_state


class PredNet(nn.Module):
    """
    Complete PredNet architecture for video prediction.

    Implements predictive coding with:
    - Hierarchical predictions at multiple scales
    - Error-driven learning
    - Recurrent temporal dynamics
    """

    def __init__(self, input_channels=3, layer_channels=[64, 128, 256]):
        super().__init__()

        self.num_layers = len(layer_channels)
        self.layer_channels = layer_channels

        # Build layers
        self.layers = nn.ModuleList()

        for l in range(self.num_layers):
            if l == 0:
                in_channels = input_channels
            else:
                in_channels = layer_channels[l-1]

            repr_channels = layer_channels[l]
            error_channels = 2 * in_channels  # pos + neg errors

            self.layers.append(PredNetLayer(
                in_channels, repr_channels, error_channels
            ))

        # Downsampling/upsampling for hierarchy
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Initial error (input at first timestep)
        self.initial_error = nn.Conv2d(input_channels, 2 * input_channels, 1)

    def init_states(self, batch_size, height, width, device):
        """Initialize hidden states for all layers."""
        states = []
        h, w = height, width

        for l in range(self.num_layers):
            channels = self.layer_channels[l]
            h_state = torch.zeros(batch_size, channels, h, w, device=device)
            c_state = torch.zeros(batch_size, channels, h, w, device=device)
            states.append((h_state, c_state))
            h, w = h // 2, w // 2

        return states

    def forward(self, x_seq, teacher_forcing=True):
        """
        Forward pass through video sequence.

        Args:
            x_seq: (B, T, C, H, W) input video sequence
            teacher_forcing: Use ground truth as input (training)

        Returns:
            predictions: (B, T, C, H, W) predicted frames
            errors: List of error tensors per layer
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # Initialize states
        states = self.init_states(B, H, W, device)

        predictions = []
        all_errors = []

        for t in range(T):
            if t == 0 or teacher_forcing:
                x = x_seq[:, t]
            else:
                x = prediction  # Use previous prediction

            # Bottom-up pass: compute errors at each level
            a_targets = [x]
            e_list = []

            # Initial error for first layer
            if t == 0:
                e = self.initial_error(x)
            else:
                e = e_list[0] if len(e_list) > 0 else self.initial_error(x)

            # Forward through layers bottom-up
            representations = []
            new_states = []

            for l in range(self.num_layers):
                # Get target for this layer
                if l == 0:
                    a_target = x
                else:
                    a_target = self.downsample(representations[l-1])

                # Get representation from above (top-down)
                if l < self.num_layers - 1 and len(representations) > l:
                    r_above = self.upsample(representations[l+1]) if len(representations) > l+1 else None
                else:
                    r_above = None

                # Update layer
                r, e, a_hat, new_state = self.layers[l](
                    a_target, e, r_above, states[l]
                )

                representations.append(r)
                e_list.append(e)
                new_states.append(new_state)

                # Downsample error for next layer
                if l < self.num_layers - 1:
                    e = self.downsample(e)

            # Update states
            states = new_states

            # Top-down pass to refine predictions
            prediction = self.layers[0].prediction(representations[0])
            predictions.append(prediction)
            all_errors.append(e_list)

        predictions = torch.stack(predictions, dim=1)

        return predictions, all_errors

    def predict_future(self, context_frames, n_future):
        """
        Predict future frames given context.

        Args:
            context_frames: (B, T_context, C, H, W) context frames
            n_future: Number of future frames to predict
        """
        B, T_context, C, H, W = context_frames.shape
        device = context_frames.device

        # Process context with teacher forcing
        states = self.init_states(B, H, W, device)

        # Run through context
        for t in range(T_context):
            x = context_frames[:, t]
            # ... (same as forward, but just update states)

        # Predict future without teacher forcing
        future_predictions = []
        prediction = context_frames[:, -1]  # Start from last context frame

        for t in range(n_future):
            # Run single step without teacher forcing
            prediction, _ = self._step(prediction, states)
            future_predictions.append(prediction)

        return torch.stack(future_predictions, dim=1)


def prednet_loss(predictions, targets, errors, error_weights=None):
    """
    PredNet loss combining reconstruction and error terms.

    Args:
        predictions: Predicted frames
        targets: Ground truth frames
        errors: Error tensors from all layers
        error_weights: Weights for each layer's error
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(predictions, targets)

    # Error loss (minimize prediction errors at all levels)
    if error_weights is None:
        error_weights = [1.0] * len(errors[0])

    error_loss = 0
    for t_errors in errors:
        for l, e in enumerate(t_errors):
            error_loss += error_weights[l] * e.mean()

    return recon_loss + 0.1 * error_loss
```

---

## 2. PredRNN: Spatiotemporal LSTMs for Video Prediction

### 2.1 Spatiotemporal Memory Flow

PredRNN (Wang et al., 2017) introduces a novel memory mechanism that enables information to flow both vertically (across layers) and horizontally (across time):

From [PredRNN: Recurrent Neural Networks for Predictive Learning](https://github.com/thuml/predrnn-pytorch):

**Key Innovation - Zigzag Memory Flow:**

```
Standard LSTM: Memory flows only horizontally (time)
    M_t^l -> M_{t+1}^l

PredRNN: Memory flows both directions
    M_t^l -> M_t^{l+1}     (bottom-up within timestep)
    M_t^L -> M_{t+1}^1     (top to bottom across time)
```

### 2.2 PyTorch SpatioTemporal LSTM Implementation

```python
import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    """
    Spatiotemporal LSTM cell with dual memory states.

    Key insight: Separate temporal memory (C) from spatiotemporal memory (M)
    - C: Standard LSTM cell state (temporal)
    - M: Zigzag memory (spatiotemporal)
    """

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Gates for temporal memory (standard LSTM)
        self.conv_x = nn.Conv2d(in_channels, 7 * hidden_channels, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

        # Gates for spatiotemporal memory
        self.conv_m = nn.Conv2d(hidden_channels, 3 * hidden_channels, kernel_size, padding=padding)

        # Output gate combines both memories
        self.conv_o = nn.Conv2d(2 * hidden_channels, hidden_channels, 1)

    def forward(self, x, h, c, m):
        """
        Args:
            x: Input at current timestep
            h: Hidden state from same layer, previous timestep
            c: Cell state from same layer, previous timestep
            m: Spatiotemporal memory (zigzag flow)

        Returns:
            h_new: New hidden state
            c_new: New cell state
            m_new: New spatiotemporal memory
        """
        # Gates from input
        x_gates = self.conv_x(x)
        x_i, x_f, x_g, x_i_prime, x_f_prime, x_g_prime, x_o = torch.chunk(x_gates, 7, dim=1)

        # Gates from hidden state
        h_gates = self.conv_h(h)
        h_i, h_f, h_g, h_o = torch.chunk(h_gates, 4, dim=1)

        # Gates from spatiotemporal memory
        m_gates = self.conv_m(m)
        m_i, m_f, m_g = torch.chunk(m_gates, 3, dim=1)

        # Temporal memory update (standard LSTM)
        i = torch.sigmoid(x_i + h_i)
        f = torch.sigmoid(x_f + h_f)
        g = torch.tanh(x_g + h_g)
        c_new = f * c + i * g

        # Spatiotemporal memory update
        i_prime = torch.sigmoid(x_i_prime + m_i)
        f_prime = torch.sigmoid(x_f_prime + m_f)
        g_prime = torch.tanh(x_g_prime + m_g)
        m_new = f_prime * m + i_prime * g_prime

        # Output gate combines both memories
        o = torch.sigmoid(x_o + h_o + self.conv_o(torch.cat([c_new, m_new], dim=1)))

        # Hidden state from both memories
        h_new = o * torch.tanh(torch.cat([c_new, m_new], dim=1))
        h_new = nn.Conv2d(2 * self.hidden_channels, self.hidden_channels, 1).to(x.device)(h_new)

        return h_new, c_new, m_new


class PredRNN(nn.Module):
    """
    PredRNN for spatiotemporal predictive learning.

    Features:
    - Zigzag memory flow across layers and time
    - Spatiotemporal LSTM cells
    - Multi-scale temporal dynamics
    """

    def __init__(self, input_channels=1, hidden_channels=[64, 64, 64, 64],
                 kernel_size=3, img_size=(64, 64)):
        super().__init__()

        self.num_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels
        self.img_size = img_size

        # Encoder
        self.encoder = nn.Conv2d(input_channels, hidden_channels[0], kernel_size, padding=1)

        # Spatiotemporal LSTM layers
        self.st_lstm_layers = nn.ModuleList()
        for l in range(self.num_layers):
            in_ch = hidden_channels[l-1] if l > 0 else hidden_channels[0]
            self.st_lstm_layers.append(
                SpatioTemporalLSTMCell(in_ch, hidden_channels[l], kernel_size)
            )

        # Decoder
        self.decoder = nn.Conv2d(hidden_channels[-1], input_channels, kernel_size, padding=1)

    def init_states(self, batch_size, device):
        """Initialize hidden, cell, and spatiotemporal memory states."""
        states = []
        H, W = self.img_size

        for l in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_channels[l], H, W, device=device)
            c = torch.zeros(batch_size, self.hidden_channels[l], H, W, device=device)
            states.append((h, c))

        # Initial spatiotemporal memory
        m = torch.zeros(batch_size, self.hidden_channels[0], H, W, device=device)

        return states, m

    def forward(self, x_seq, n_future=0, teacher_forcing_ratio=1.0):
        """
        Forward pass through sequence.

        Args:
            x_seq: (B, T, C, H, W) input sequence
            n_future: Number of future frames to predict
            teacher_forcing_ratio: Probability of using ground truth
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # Initialize states
        states, m = self.init_states(B, device)

        outputs = []

        for t in range(T + n_future):
            # Get input
            if t < T:
                if t == 0 or torch.rand(1).item() < teacher_forcing_ratio:
                    x = x_seq[:, t]
                else:
                    x = output
            else:
                x = output  # Use prediction for future frames

            # Encode input
            x = self.encoder(x)

            # Forward through ST-LSTM layers with zigzag memory
            for l in range(self.num_layers):
                h, c = states[l]

                # Input to this layer
                if l == 0:
                    layer_input = x
                else:
                    layer_input = states[l-1][0]  # Hidden from layer below

                # ST-LSTM forward
                h_new, c_new, m = self.st_lstm_layers[l](layer_input, h, c, m)
                states[l] = (h_new, c_new)

            # Decode output
            output = self.decoder(states[-1][0])
            outputs.append(output)

        return torch.stack(outputs, dim=1)


class PredRNNv2(PredRNN):
    """
    PredRNN-V2 with memory decoupling and reverse scheduled sampling.

    Improvements over PredRNN:
    1. Memory decoupling loss
    2. Reverse scheduled sampling
    3. Better long-term prediction
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Decoupling loss projection
        self.decouple_proj = nn.ModuleList([
            nn.Conv2d(ch, ch, 1) for ch in self.hidden_channels
        ])

    def forward(self, x_seq, n_future=0, teacher_forcing_ratio=1.0,
                reverse_scheduled=False):
        """Forward with optional reverse scheduled sampling."""

        outputs = super().forward(x_seq, n_future, teacher_forcing_ratio)

        return outputs

    def memory_decoupling_loss(self, states):
        """
        Loss to encourage C and M to learn different features.

        Minimizes similarity between cell state C and memory M.
        """
        loss = 0
        for l, (h, c) in enumerate(states):
            # Project C to same space
            c_proj = self.decouple_proj[l](c)

            # Cosine similarity loss
            c_flat = c_proj.view(c_proj.size(0), -1)
            m_flat = h.view(h.size(0), -1)  # Using h as proxy for m

            similarity = F.cosine_similarity(c_flat, m_flat, dim=1)
            loss += similarity.mean()

        return loss / len(states)
```

---

## 3. Video Prediction Through Predictive Coding

### 3.1 Why Prediction is a Good Objective

From [Deep Predictive Coding Networks](https://arxiv.org/abs/1605.08104):

**Theoretical Motivation:**

```
Prediction as self-supervision:
- No labels needed - future frames are the labels
- Forces model to learn causal structure
- Naturally captures temporal dynamics
- Generalizes to novel situations

Information-theoretic view:
- Good prediction requires good representation
- Prediction error = Information to be learned
- Minimizing prediction error = Compression
```

### 3.2 Multi-Scale Temporal Processing

```python
class MultiScalePredictiveNetwork(nn.Module):
    """
    Multi-scale temporal prediction with different timescales.

    Different layers operate at different temporal resolutions:
    - Fast layers: Frame-to-frame motion
    - Medium layers: Action segments
    - Slow layers: Scene dynamics
    """

    def __init__(self, channels=[64, 128, 256], timescales=[1, 2, 4]):
        super().__init__()

        self.num_scales = len(channels)
        self.timescales = timescales

        # Build predictive modules at each scale
        self.predictors = nn.ModuleList()
        self.update_rates = nn.ModuleList()

        for i in range(self.num_scales):
            # Predictor at this scale
            self.predictors.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i], 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels[i], channels[i], 3, padding=1)
            ))

            # Learnable update rate (how fast this scale updates)
            self.update_rates.append(nn.Parameter(
                torch.ones(1) * (1.0 / timescales[i])
            ))

        # Cross-scale connections
        self.top_down = nn.ModuleList([
            nn.Conv2d(channels[i+1], channels[i], 1)
            for i in range(self.num_scales - 1)
        ])

        self.bottom_up = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1)
            for i in range(self.num_scales - 1)
        ])

    def forward(self, features_list, timestep):
        """
        Update predictions at multiple timescales.

        Args:
            features_list: Features at each scale
            timestep: Current timestep
        """
        predictions = []
        errors = []

        for i in range(self.num_scales):
            # Check if this scale should update
            if timestep % self.timescales[i] == 0:
                # Generate prediction
                pred = self.predictors[i](features_list[i])

                # Add top-down prediction if available
                if i < self.num_scales - 1 and len(predictions) > i + 1:
                    top_down_pred = F.interpolate(
                        self.top_down[i](predictions[i+1]),
                        size=pred.shape[-2:]
                    )
                    pred = pred + top_down_pred

                predictions.append(pred)

                # Compute error
                error = features_list[i] - pred
                errors.append(error)
            else:
                # Keep previous prediction
                predictions.append(None)
                errors.append(None)

        return predictions, errors
```

---

## 4. Temporal Unrolling and Training Strategies

### 4.1 Backpropagation Through Time (BPTT)

```python
def train_prednet_bptt(model, dataloader, optimizer, n_epochs,
                       sequence_length=20, tbptt_length=10):
    """
    Train PredNet with truncated BPTT for long sequences.

    Args:
        model: PredNet model
        dataloader: Video sequence dataloader
        optimizer: Optimizer
        n_epochs: Number of epochs
        sequence_length: Full sequence length
        tbptt_length: Truncation length for BPTT
    """
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0

        for batch in dataloader:
            videos = batch['video']  # (B, T, C, H, W)
            B, T, C, H, W = videos.shape

            # Initialize hidden states
            states = model.init_states(B, H, W, videos.device)

            total_loss = 0

            # Truncated BPTT: process in chunks
            for start in range(0, T - 1, tbptt_length):
                end = min(start + tbptt_length, T - 1)

                # Get chunk
                chunk = videos[:, start:end+1]

                # Detach states for truncation
                states = [(h.detach(), c.detach()) for h, c in states]

                # Forward pass
                predictions, errors = model(chunk, teacher_forcing=True)

                # Loss on this chunk
                targets = chunk[:, 1:]  # Predict next frame
                loss = prednet_loss(predictions[:, :-1], targets, errors)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()

            epoch_loss += total_loss

        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}")


def scheduled_sampling(model, videos, optimizer, epoch, max_epochs):
    """
    Scheduled sampling: Gradually reduce teacher forcing.

    Start with 100% teacher forcing, decrease over training.
    """
    # Inverse sigmoid schedule
    k = 5.0
    teacher_forcing_ratio = k / (k + torch.exp(torch.tensor(epoch / max_epochs * 10)))

    predictions, errors = model(videos, teacher_forcing_ratio=teacher_forcing_ratio.item())

    return predictions, errors


def reverse_scheduled_sampling(model, videos, optimizer, epoch, max_epochs):
    """
    Reverse scheduled sampling (PredRNN-V2).

    For ENCODER: Start with predictions, move to ground truth
    For DECODER: Standard scheduled sampling

    This forces model to learn to use context even when
    previous predictions are wrong.
    """
    k = 5.0

    # Encoder: reverse schedule (0 -> 1 teacher forcing)
    encoder_tf = 1 - k / (k + torch.exp(torch.tensor(epoch / max_epochs * 10)))

    # Decoder: standard schedule (1 -> 0 teacher forcing)
    decoder_tf = k / (k + torch.exp(torch.tensor(epoch / max_epochs * 10)))

    # Apply different ratios to encoder/decoder
    predictions = model(
        videos,
        encoder_tf_ratio=encoder_tf.item(),
        decoder_tf_ratio=decoder_tf.item()
    )

    return predictions
```

### 4.2 Training Loop with Multiple Loss Terms

```python
def train_video_prediction(model, train_loader, val_loader, config):
    """Complete training loop for video prediction models."""

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0

        for batch in train_loader:
            videos = batch['video'].to(config['device'])

            # Forward pass
            predictions, errors = model(videos)

            # Multi-component loss
            loss = compute_video_prediction_loss(
                predictions, videos, errors, config
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                videos = batch['video'].to(config['device'])
                predictions, errors = model(videos)
                loss = compute_video_prediction_loss(
                    predictions, videos, errors, config
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.4f}, "
              f"Val={val_loss:.4f}")


def compute_video_prediction_loss(predictions, targets, errors, config):
    """
    Multi-component loss for video prediction.

    Components:
    1. MSE reconstruction
    2. Perceptual loss (VGG features)
    3. Adversarial loss (optional)
    4. Error minimization (predictive coding)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(predictions[:, :-1], targets[:, 1:])

    # Perceptual loss (if using)
    if config.get('use_perceptual', False):
        perceptual_loss = compute_perceptual_loss(
            predictions[:, :-1], targets[:, 1:], config['vgg']
        )
    else:
        perceptual_loss = 0

    # Error loss (predictive coding)
    error_loss = sum(e.abs().mean() for t_errors in errors for e in t_errors)

    # Combine losses
    total_loss = (
        config['lambda_recon'] * recon_loss +
        config['lambda_perceptual'] * perceptual_loss +
        config['lambda_error'] * error_loss
    )

    return total_loss


def compute_perceptual_loss(pred, target, vgg_model):
    """Perceptual loss using VGG features."""
    # Get VGG features
    pred_features = vgg_model(pred.view(-1, *pred.shape[-3:]))
    target_features = vgg_model(target.view(-1, *target.shape[-3:]))

    # L1 loss on features
    loss = F.l1_loss(pred_features, target_features)

    return loss
```

---

## 5. Performance Optimization

### 5.1 Memory-Efficient Training

```python
class GradientCheckpointedPredNet(PredNet):
    """
    PredNet with gradient checkpointing for memory efficiency.

    Trades compute for memory: recompute activations during backward.
    """

    def forward(self, x_seq, teacher_forcing=True):
        """Forward with gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint

        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        states = self.init_states(B, H, W, device)
        predictions = []

        for t in range(T):
            x = x_seq[:, t]

            # Checkpoint each timestep
            def forward_step(x, states):
                # ... single timestep forward
                return prediction, new_states, errors

            prediction, states, errors = checkpoint(
                forward_step, x, states, use_reentrant=False
            )

            predictions.append(prediction)

        return torch.stack(predictions, dim=1)


class EfficientConvLSTM(nn.Module):
    """
    Memory-efficient ConvLSTM using depthwise separable convolutions.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(
            input_channels + hidden_channels,
            input_channels + hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=input_channels + hidden_channels
        )

        self.pointwise = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            1
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)

        # Depthwise separable convolution
        out = self.depthwise(combined)
        gates = self.pointwise(out)

        # Split gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)

        return h_new, (h_new, c_new)
```

### 5.2 CUDA Optimization Tips

```python
# Performance tips for video prediction models:

# 1. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    predictions, errors = model(videos)
    loss = compute_loss(predictions, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()


# 2. Efficient data loading
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2
)


# 3. Compile model (PyTorch 2.0+)
model = torch.compile(model, mode='reduce-overhead')


# 4. Memory format optimization
model = model.to(memory_format=torch.channels_last)
videos = videos.to(memory_format=torch.channels_last)


# 5. Benchmarking
torch.backends.cudnn.benchmark = True
```

---

## 6. TRAIN STATION: Recurrent = Temporal = Sequence = Memory

### The Grand Unification

**THE TRAIN STATION** where multiple concepts meet:

```
        RECURRENT
           |
           |
    ┌──────┼──────┐
    |      |      |
 TEMPORAL  |   SEQUENCE
    |      |      |
    └──────┼──────┘
           |
         MEMORY
           |
           ▼
    PREDICTIVE CODING
```

### 6.1 The Topological Equivalence

**Recurrent = Temporal Processing**

```python
# RNN is just unrolled temporal processing
def rnn_as_temporal(inputs, initial_state, cell):
    """RNN = repeated application of same function over time"""
    state = initial_state
    outputs = []

    for t in range(len(inputs)):
        output, state = cell(inputs[t], state)
        outputs.append(output)

    return outputs, state

# This IS temporal processing:
# - Same computation applied repeatedly
# - State carries information through time
# - Order matters (causality)
```

**Temporal = Sequence Modeling**

```python
# Sequence = ordered collection with dependencies
# Temporal = time-indexed sequence

# They're the same thing!
sequence_data = [x_0, x_1, x_2, ..., x_T]
temporal_data = {t: x_t for t in range(T)}

# Both capture:
# - Ordering/causality
# - Dependencies between elements
# - History matters for current state
```

**Sequence = Memory**

```python
# Memory IS the encoding of past sequences
class SequenceMemory(nn.Module):
    """Sequence modeling = memory compression"""

    def encode_sequence(self, sequence):
        # Process sequence
        for x in sequence:
            self.memory = self.update(self.memory, x)
        return self.memory

    def decode_from_memory(self, memory):
        # Generate sequence from memory
        outputs = []
        state = memory
        for _ in range(length):
            output, state = self.generate(state)
            outputs.append(output)
        return outputs

# Memory = compressed representation of sequence
# Sequence = unfolded memory
```

**Memory = Predictive Coding**

```python
# Predictive coding uses memory to predict future
class PredictiveMemory(nn.Module):
    """Memory enables prediction"""

    def forward(self, observation):
        # Use memory to predict
        prediction = self.predict(self.memory)

        # Compute error
        error = observation - prediction

        # Update memory based on error
        self.memory = self.update(self.memory, error)

        return prediction, error

# Good memory -> good predictions -> small errors
# This IS what brains do!
```

### 6.2 The Mathematical Unity

```
All are solving the same problem:

minimize E[||x_{t+1} - f(x_1, ..., x_t)||^2]

- Recurrent: f is parameterized by recurrent weights
- Temporal: f captures temporal dynamics
- Sequence: f models sequence dependencies
- Memory: f accesses compressed history
- Predictive Coding: f generates predictions, updates on errors

SAME OBJECTIVE, DIFFERENT NAMES!
```

### 6.3 Train Station Connections

**Connection to Transformers:**

```python
# Transformer attention = soft addressing of memory
# RNN memory = hard compression of history

class UnifiedSequenceModel(nn.Module):
    """Both transformers and RNNs are sequence models"""

    def forward_rnn(self, x_seq):
        # Compress history into fixed-size state
        memory = self.initial_state
        for x in x_seq:
            memory = self.update(memory, x)
        return memory

    def forward_transformer(self, x_seq):
        # Attend over full history
        memory = x_seq  # Keep all history
        output = self.attend(query, memory)
        return output

    # Both model sequences!
    # RNN: O(1) memory, O(T) compute per step
    # Transformer: O(T) memory, O(T^2) compute per step
```

**Connection to State Space Models:**

```python
# SSM (Mamba, S4) = continuous-time RNN
# dx/dt = Ax + Bu
# y = Cx + Du

# Discretized:
# x_t = A_d * x_{t-1} + B_d * u_t
# y_t = C * x_t + D * u_t

# This IS an RNN with structured transition matrix!
# SSM = efficient RNN with structured memory
```

**Connection to Active Inference:**

```python
# Active inference: minimize free energy over time
# Recurrent prediction: minimize prediction error over time

# Both:
# 1. Maintain beliefs/states about world
# 2. Generate predictions
# 3. Update based on errors
# 4. Use temporal structure

# Recurrent predictive networks ARE
# a form of active inference!
```

---

## 7. ARR-COC-0-1 Connection: Temporal Relevance Prediction

### 7.1 Relevance as Temporal Prediction

In ARR-COC-0-1, we can use recurrent predictive networks for temporal token allocation:

```python
class TemporalRelevancePredictor(nn.Module):
    """
    Predict which tokens will be relevant based on temporal context.

    Key insight: Relevance follows temporal patterns:
    - Objects that were relevant stay relevant
    - Attention flows along motion paths
    - Context builds over time
    """

    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()

        # Temporal memory for relevance patterns
        self.relevance_lstm = nn.LSTMCell(feature_dim, hidden_dim)

        # Predict future relevance
        self.relevance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Error-based adaptation (predictive coding)
        self.error_processor = nn.Linear(1, hidden_dim)

    def forward(self, token_features, prev_relevance, state):
        """
        Args:
            token_features: Current token features (N, D)
            prev_relevance: Previous relevance scores (N,)
            state: LSTM state

        Returns:
            predicted_relevance: For current timestep
            future_relevance: Prediction for next timestep
        """
        N, D = token_features.shape

        predictions = []

        for i in range(N):
            # Update temporal memory
            h, c = self.relevance_lstm(token_features[i:i+1], state)
            state = (h, c)

            # Predict relevance
            pred = self.relevance_predictor(h)
            predictions.append(pred)

            # Compute prediction error
            if prev_relevance is not None:
                error = prev_relevance[i:i+1].unsqueeze(-1) - pred
                # Incorporate error into state
                error_signal = self.error_processor(error)
                h = h + 0.1 * error_signal

        predicted_relevance = torch.cat(predictions, dim=0).squeeze(-1)

        return predicted_relevance, state


class RecurrentRelevanceAllocator(nn.Module):
    """
    Allocate compute tokens using recurrent prediction.

    The allocator learns:
    - Which tokens were relevant before
    - How relevance changes over time
    - Predict where to allocate next
    """

    def __init__(self, vision_encoder, temporal_predictor):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.temporal_predictor = temporal_predictor

    def forward(self, video_frames, token_budget):
        """
        Allocate tokens across video frames.

        Args:
            video_frames: (B, T, C, H, W) video
            token_budget: Total tokens to allocate
        """
        B, T, C, H, W = video_frames.shape

        allocations = []
        states = None
        prev_relevance = None

        for t in range(T):
            # Encode frame
            features = self.vision_encoder(video_frames[:, t])

            # Predict relevance
            relevance, states = self.temporal_predictor(
                features, prev_relevance, states
            )

            # Allocate tokens based on predicted relevance
            allocation = self.allocate_tokens(
                relevance, token_budget // T
            )

            allocations.append(allocation)
            prev_relevance = relevance.detach()

        return allocations

    def allocate_tokens(self, relevance, budget):
        """Top-k allocation based on relevance."""
        k = min(budget, len(relevance))
        _, indices = torch.topk(relevance, k)
        return indices
```

### 7.2 Predictive Token Routing

```python
class PredictiveTokenRouter(nn.Module):
    """
    Route tokens using predictive coding principles.

    Predict which expert should process which token,
    update predictions based on errors.
    """

    def __init__(self, num_experts, feature_dim):
        super().__init__()

        self.num_experts = num_experts

        # Routing predictor (per expert)
        self.route_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1)
            )
            for _ in range(num_experts)
        ])

        # Temporal memory for routing patterns
        self.route_memory = nn.LSTMCell(num_experts, feature_dim // 2)

    def forward(self, tokens, state=None):
        """
        Route tokens to experts using predictions.

        Args:
            tokens: (N, D) token features
            state: Memory state
        """
        N, D = tokens.shape

        # Predict routing scores
        route_scores = torch.stack([
            pred(tokens).squeeze(-1)
            for pred in self.route_predictors
        ], dim=1)  # (N, num_experts)

        # Softmax for routing probabilities
        route_probs = F.softmax(route_scores, dim=1)

        # Update memory with routing decisions
        if state is not None:
            # Average routing decision
            avg_routes = route_probs.mean(dim=0, keepdim=True)
            h, c = self.route_memory(avg_routes, state)
            state = (h, c)

        # Select experts (top-k routing)
        _, expert_indices = torch.topk(route_probs, k=2, dim=1)

        return route_probs, expert_indices, state
```

### 7.3 Temporal Context for Relevance

```python
class TemporalContextRelevance(nn.Module):
    """
    Build temporal context for better relevance scoring.

    Uses PredNet-style architecture for visual relevance.
    """

    def __init__(self, input_dim, context_dim=256):
        super().__init__()

        # Context encoder (what happened before)
        self.context_encoder = nn.GRU(input_dim, context_dim, batch_first=True)

        # Relevance predictor
        self.relevance_head = nn.Sequential(
            nn.Linear(context_dim + input_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, 1),
            nn.Sigmoid()
        )

        # Prediction error tracker
        self.error_history = []

    def forward(self, token_sequence):
        """
        Score relevance with temporal context.

        Args:
            token_sequence: (B, T, N, D) tokens over time
        """
        B, T, N, D = token_sequence.shape

        # Encode context
        context_input = token_sequence.mean(dim=2)  # (B, T, D)
        context, _ = self.context_encoder(context_input)

        # Score relevance at each timestep
        relevance_scores = []

        for t in range(T):
            # Get current tokens
            current_tokens = token_sequence[:, t]  # (B, N, D)

            # Get temporal context
            temporal_context = context[:, t:t+1].expand(-1, N, -1)  # (B, N, context_dim)

            # Combine and predict relevance
            combined = torch.cat([current_tokens, temporal_context], dim=-1)
            relevance = self.relevance_head(combined).squeeze(-1)  # (B, N)

            relevance_scores.append(relevance)

        return torch.stack(relevance_scores, dim=1)  # (B, T, N)
```

---

## 8. Sources and References

### Research Papers

**PredNet and Predictive Coding:**
- [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104) - Lotter et al., 2016 (Cited 1285+)
- [PredNet and Predictive Coding: A Critical Review](https://arxiv.org/abs/1906.11902) - Rane et al., 2019 (Cited 39)
- [Predictive coding networks for temporal prediction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011183) - Millidge et al., 2024 (Cited 47)

**PredRNN and Spatiotemporal LSTMs:**
- [PredRNN: Recurrent Neural Networks for Predictive Learning Using Spatiotemporal LSTMs](https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms) - Wang et al., NeurIPS 2017
- [PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning](https://arxiv.org/abs/2103.09504) - Wang et al., TPAMI 2022

**Predictive Coding in Neural Networks:**
- [Predictive coding is a consequence of energy efficiency in recurrent neural networks](https://www.sciencedirect.com/science/article/pii/S2666389922002719) - Ali et al., 2022 (Cited 71)
- [Predictive Coding Based Multiscale Network with Encoder-Decoder LSTM for Video Prediction](https://arxiv.org/abs/2212.11642) - Ling et al., 2022

### Code Repositories

**PredNet:**
- [coxlab/prednet](https://github.com/coxlab/prednet) - Official Keras implementation
- [PredNet by coxlab](https://coxlab.github.io/prednet/) - Project page with models

**PredRNN:**
- [thuml/predrnn-pytorch](https://github.com/thuml/predrnn-pytorch) - Official PyTorch implementation (506 stars)
- Pretrained models: [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/72241e0046a74f81bf29/) / [Google Drive](https://drive.google.com/drive/folders/1jaEHcxo_UgvgwEWKi0ygX1SbODGz6PWw)

### Foundational Work

- Rao & Ballard (1999) - Predictive coding in the visual cortex
- Friston (2005) - A theory of cortical responses
- Hochreiter & Schmidhuber (1997) - Long Short-Term Memory

### Datasets

- KITTI Dataset - Self-driving video sequences
- Caltech Pedestrian Dataset - Pedestrian detection videos
- Moving MNIST - Synthetic digit motion
- KTH Action Dataset - Human action recognition
- BAIR Robot Pushing Dataset - Robotic manipulation

---

## Summary

Recurrent predictive networks represent a powerful convergence of:

1. **Neuroscience** (predictive coding theory)
2. **Deep learning** (ConvLSTMs, hierarchical representations)
3. **Temporal modeling** (sequence prediction, video forecasting)

**Key Architectures:**
- **PredNet**: Hierarchical predictive coding with error propagation
- **PredRNN**: Spatiotemporal LSTMs with zigzag memory flow
- **PredRNN-V2**: Memory decoupling + reverse scheduled sampling

**The TRAIN STATION unifies:**
- Recurrent = Temporal = Sequence = Memory
- All solve: "predict next given history"
- All use: state compression + error-driven learning

**For ARR-COC-0-1:**
- Predict temporal relevance patterns
- Route tokens based on learned dynamics
- Build context for better allocation

**The brain predicts to understand. These networks learn the same way.**
