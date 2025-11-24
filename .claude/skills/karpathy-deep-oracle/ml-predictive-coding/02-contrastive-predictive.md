# Contrastive Predictive Coding: InfoNCE Loss and Self-Supervised Learning

## Overview

Contrastive Predictive Coding (CPC) is a foundational framework for self-supervised representation learning that learns by predicting future observations in latent space. Rather than reconstructing raw pixels (expensive and often unnecessary), CPC learns to distinguish "correct" future predictions from "distractor" samples using the InfoNCE loss. This approach connects temporal prediction, mutual information maximization, and contrastive learning into a unified framework.

**Key Insight**: CPC doesn't predict future observations directly - it learns representations where the future is PREDICTABLE. This is the same principle behind CLIP, SimCLR, and modern self-supervised methods.

---

## Section 1: The CPC Framework

### Core Architecture

CPC consists of three main components working together:

```
Raw Input (x_t) --> Encoder (g_enc) --> Latent (z_t) --> Autoregressive (g_ar) --> Context (c_t)
                                                                                      |
                                                                                      v
                                                                              Predict Future z_{t+k}
```

**1. Encoder Network (g_enc)**
- Maps high-dimensional inputs x_t to latent representations z_t
- Typically a CNN for images, or strided convolutions for audio
- Compresses information while preserving structure

**2. Autoregressive Model (g_ar)**
- Summarizes past latents z_{<=t} into a context vector c_t
- Often a GRU, LSTM, or causal transformer
- c_t captures what's happened so far

**3. Prediction Network (W_k)**
- For each future step k, learns a linear transformation W_k
- Predicts z_{t+k} from context c_t
- Different W_k for different prediction horizons

### The Prediction Task

Given context c_t, predict the latent representation z_{t+k} for future time step t+k:

```python
# Predicted future representation
z_hat_future = W_k @ c_t  # Linear prediction

# Actual future representation
z_future = encoder(x_{t+k})  # Encode actual future observation

# Goal: Make z_hat_future similar to z_future
# But HOW do we define "similar"?
```

The genius of CPC: Don't define similarity directly - use CONTRASTIVE learning!

---

## Section 2: InfoNCE Loss - The Heart of CPC

### The Fundamental Idea

InfoNCE (Information Noise Contrastive Estimation) frames prediction as a classification problem:

**Given:** Context c_t and K+1 candidate future observations
**Task:** Identify which one is the TRUE future among K distractors

This is a (K+1)-way classification problem with a categorical cross-entropy loss.

### Mathematical Formulation

The InfoNCE loss for a single positive pair (z, c) with N-1 negative samples:

```
L_InfoNCE = -log[ exp(f(z, c) / tau) / sum_{i=1}^{N} exp(f(z_i, c) / tau) ]
```

Where:
- f(z, c) is a scoring function (typically dot product: z^T W_k c)
- tau is temperature parameter
- z is the true positive (correct future)
- z_i includes the positive and N-1 negatives

### Expanded Form

```
L_InfoNCE = -log[ exp(z^T W_k c / tau) / (exp(z^T W_k c / tau) + sum_{j} exp(z_j^T W_k c / tau)) ]
```

This is exactly a softmax cross-entropy loss where:
- The "logit" for the positive is z^T W_k c
- The "logits" for negatives are z_j^T W_k c

---

## Section 3: InfoNCE as Mutual Information Lower Bound

### The Information-Theoretic Connection

The key theoretical insight: minimizing InfoNCE maximizes a lower bound on mutual information I(X; C).

**Mutual Information Definition:**
```
I(X; C) = E_{p(x,c)}[log(p(x|c) / p(x))]
```

This measures how much knowing C tells us about X.

### Derivation of the Lower Bound

Starting from the InfoNCE objective, we can show:

```
I(X; C) >= log(N) - L_InfoNCE
```

Where N is the number of samples (1 positive + N-1 negatives).

**Key Steps in Derivation:**

1. **Density Ratio Trick**: The optimal scoring function f*(x, c) equals:
   ```
   f*(x, c) proportional to p(x|c) / p(x)
   ```

2. **Jensen's Inequality**: The expectation of log is bounded:
   ```
   E[log(f(x,c) / mean_f)] >= log(E[f(x,c)]) - log(E[f])
   ```

3. **The Bound**: When N approaches infinity, InfoNCE estimates MI exactly.
   For finite N: I(X; C) >= log(N) - L_InfoNCE

### The log(N) Ceiling

**Critical Limitation**: InfoNCE can only estimate MI up to log(N).

If true MI is 10 nats but you only use N=100 negatives:
- Maximum estimable MI = log(100) = 4.6 nats
- Your estimate is CAPPED at 4.6, regardless of true MI

**Implications:**
- More negatives = tighter bound (but more compute)
- For high-MI problems, you need MANY negatives
- This led to developments like SimCLR using batch size 8192

---

## Section 4: PyTorch Implementation of CPC

### Complete CPC Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CPCEncoder(nn.Module):
    """Encoder network for CPC - maps inputs to latent representations"""

    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        # Example: Simple CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CPCAutoregressive(nn.Module):
    """Autoregressive model that summarizes past into context"""

    def __init__(self, latent_dim: int, context_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=context_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, z_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_sequence: [batch, seq_len, latent_dim]
        Returns:
            contexts: [batch, seq_len, context_dim]
            hidden: final hidden state
        """
        contexts, hidden = self.gru(z_sequence)
        return contexts, hidden


class CPCModel(nn.Module):
    """Complete Contrastive Predictive Coding model"""

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 256,
        context_dim: int = 256,
        prediction_steps: int = 5,
        temperature: float = 0.07
    ):
        super().__init__()

        self.encoder = CPCEncoder(input_channels, latent_dim)
        self.autoregressive = CPCAutoregressive(latent_dim, context_dim)

        # Prediction networks for each future step
        self.prediction_heads = nn.ModuleList([
            nn.Linear(context_dim, latent_dim, bias=False)
            for _ in range(prediction_steps)
        ])

        self.prediction_steps = prediction_steps
        self.temperature = temperature
        self.latent_dim = latent_dim

    def encode_sequence(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of observations

        Args:
            x_sequence: [batch, seq_len, C, H, W]
        Returns:
            z_sequence: [batch, seq_len, latent_dim]
        """
        batch_size, seq_len = x_sequence.shape[:2]
        # Flatten batch and sequence for encoding
        x_flat = x_sequence.view(-1, *x_sequence.shape[2:])
        z_flat = self.encoder(x_flat)
        z_sequence = z_flat.view(batch_size, seq_len, -1)
        return z_sequence

    def forward(
        self,
        x_sequence: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x_sequence: [batch, seq_len, C, H, W]
        Returns:
            loss: InfoNCE loss averaged over all prediction steps
        """
        batch_size, seq_len = x_sequence.shape[:2]

        # Step 1: Encode all observations
        z_sequence = self.encode_sequence(x_sequence)  # [B, T, D]

        if return_embeddings:
            return z_sequence

        # Step 2: Get context from autoregressive model
        contexts, _ = self.autoregressive(z_sequence)  # [B, T, D]

        # Step 3: Compute InfoNCE loss for each prediction step
        total_loss = 0.0
        num_predictions = 0

        for k in range(1, self.prediction_steps + 1):
            # Context at time t predicts z at time t+k
            # We use contexts from t=0 to t=seq_len-k-1
            if seq_len - k < 1:
                continue

            c_t = contexts[:, :-k, :]  # [B, T-k, D]
            z_future = z_sequence[:, k:, :]  # [B, T-k, D]

            # Predict future latents
            z_pred = self.prediction_heads[k-1](c_t)  # [B, T-k, D]

            # Compute InfoNCE loss
            loss = self.info_nce_loss(z_pred, z_future)
            total_loss += loss
            num_predictions += 1

        return total_loss / num_predictions if num_predictions > 0 else total_loss

    def info_nce_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss

        Args:
            predictions: [batch, num_samples, dim]
            targets: [batch, num_samples, dim]
        Returns:
            loss: scalar
        """
        batch_size, num_samples, dim = predictions.shape

        # Flatten to treat each sample independently
        predictions = predictions.reshape(-1, dim)  # [B*S, D]
        targets = targets.reshape(-1, dim)  # [B*S, D]

        # Normalize for cosine similarity
        predictions = F.normalize(predictions, dim=-1)
        targets = F.normalize(targets, dim=-1)

        # Compute similarity matrix
        # Each prediction is compared with all targets
        logits = predictions @ targets.T / self.temperature  # [B*S, B*S]

        # Labels: diagonal elements are positives
        labels = torch.arange(logits.size(0), device=logits.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class InfoNCELoss(nn.Module):
    """Standalone InfoNCE loss module for general contrastive learning"""

    def __init__(
        self,
        temperature: float = 0.07,
        negative_mode: str = 'unpaired'
    ):
        super().__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode

    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, dim]
            positive_key: [batch, dim]
            negative_keys:
                - If None: use other samples in batch as negatives
                - If unpaired: [num_neg, dim]
                - If paired: [batch, num_neg, dim]
        Returns:
            loss: scalar
        """
        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)

        # Positive logits: [batch, 1]
        positive_logit = torch.sum(query * positive_key, dim=-1, keepdim=True)

        if negative_keys is None:
            # Use other samples in batch as negatives
            # [batch, batch] similarity matrix
            negative_logits = query @ positive_key.T
            # Remove diagonal (self-similarity)
            batch_size = query.size(0)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=query.device)
            negative_logits = negative_logits[mask].view(batch_size, -1)
        else:
            negative_keys = F.normalize(negative_keys, dim=-1)

            if self.negative_mode == 'unpaired':
                # Same negatives for all queries: [num_neg, dim]
                # [batch, num_neg]
                negative_logits = query @ negative_keys.T
            else:
                # Paired negatives: [batch, num_neg, dim]
                # [batch, num_neg]
                negative_logits = torch.bmm(
                    query.unsqueeze(1),
                    negative_keys.transpose(1, 2)
                ).squeeze(1)

        # Concatenate positive and negative logits
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        logits = logits / self.temperature

        # Labels: positive is always at index 0
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)

        return F.cross_entropy(logits, labels)


# Example usage
def train_cpc_example():
    """Example training loop for CPC"""

    # Model
    model = CPCModel(
        input_channels=3,
        latent_dim=256,
        context_dim=256,
        prediction_steps=5,
        temperature=0.07
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Simulated video batch: [batch, seq_len, C, H, W]
    batch_size, seq_len = 32, 16
    video_batch = torch.randn(batch_size, seq_len, 3, 64, 64)

    # Training step
    model.train()
    optimizer.zero_grad()

    loss = model(video_batch)
    loss.backward()
    optimizer.step()

    print(f"CPC Loss: {loss.item():.4f}")

    # Get learned representations
    with torch.no_grad():
        embeddings = model(video_batch, return_embeddings=True)
        print(f"Learned embeddings shape: {embeddings.shape}")

    return model


if __name__ == "__main__":
    train_cpc_example()
```

---

## Section 5: CLIP - CPC Extended to Vision-Language

### CLIP Architecture

CLIP (Contrastive Language-Image Pre-training) extends the CPC framework to multimodal learning:

```
Image --> Vision Encoder --> Image Embedding
                                    |
                                    v
                            Contrastive Loss (InfoNCE)
                                    ^
                                    |
Text --> Text Encoder --> Text Embedding
```

Instead of predicting future frames, CLIP learns to align image-text pairs.

### CLIP Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel

class CLIPModel(nn.Module):
    """Simplified CLIP implementation with InfoNCE loss"""

    def __init__(
        self,
        embed_dim: int = 512,
        temperature: float = 0.07,
        learnable_temp: bool = True
    ):
        super().__init__()

        # Vision encoder (ResNet-50)
        resnet = resnet50(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_proj = nn.Linear(2048, embed_dim)

        # Text encoder (transformer-based)
        # Using a simple transformer for demonstration
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        self.text_embedding = nn.Embedding(50000, 512)
        self.text_proj = nn.Linear(512, embed_dim)

        # Temperature parameter
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/temperature)))
        else:
            self.register_buffer('logit_scale', torch.log(torch.tensor(1/temperature)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings"""
        features = self.vision_encoder(images).squeeze(-1).squeeze(-1)
        embeddings = self.vision_proj(features)
        return F.normalize(embeddings, dim=-1)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text to embeddings"""
        # [batch, seq_len, dim]
        embedded = self.text_embedding(text_tokens)
        # Transformer expects [seq_len, batch, dim]
        features = self.text_encoder(embedded.transpose(0, 1))
        # Take [CLS] token or mean pooling
        pooled = features.mean(dim=0)
        embeddings = self.text_proj(pooled)
        return F.normalize(embeddings, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute CLIP loss (symmetric InfoNCE)"""

        # Encode both modalities
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Symmetric InfoNCE loss
        batch_size = images.size(0)
        labels = torch.arange(batch_size, device=images.device)

        # Image-to-text loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        # Text-to-image loss
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        # Total loss
        loss = (loss_i2t + loss_t2i) / 2

        return loss

    def compute_similarity(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity scores for inference"""
        with torch.no_grad():
            image_features = self.encode_image(images)
            text_features = self.encode_text(text_tokens)
            similarity = image_features @ text_features.T
        return similarity


# Zero-shot classification with CLIP
def zero_shot_classify(
    model: CLIPModel,
    images: torch.Tensor,
    class_prompts: list,
    tokenizer
) -> torch.Tensor:
    """Zero-shot image classification using CLIP

    Args:
        model: Trained CLIP model
        images: [batch, C, H, W]
        class_prompts: List of text prompts for each class
            e.g., ["a photo of a dog", "a photo of a cat", ...]
        tokenizer: Text tokenizer
    Returns:
        predictions: [batch] class indices
    """
    model.eval()

    # Tokenize class prompts
    text_tokens = tokenizer(
        class_prompts,
        padding=True,
        return_tensors="pt"
    )["input_ids"]

    # Compute similarity
    similarity = model.compute_similarity(images, text_tokens)

    # Predict most similar class
    predictions = similarity.argmax(dim=-1)

    return predictions
```

---

## Section 6: The Train Station - Where CPC Meets Everything

### The Grand Unification: CPC = CLIP = SimCLR = Active Inference

**TRAIN STATION**: All contrastive methods are doing the SAME thing!

```
                    ┌─────────────────────────────┐
                    │   CONTRASTIVE PREDICTION    │
                    │   = Mutual Information      │
                    │   = Free Energy Minimization│
                    └──────────────┬──────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           v                       v                       v
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │    CPC      │         │    CLIP     │         │   SimCLR    │
    │             │         │             │         │             │
    │ Predict     │         │ Align       │         │ Align       │
    │ Future      │         │ Image-Text  │         │ Augmented   │
    │ Frames      │         │ Pairs       │         │ Views       │
    └─────────────┘         └─────────────┘         └─────────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                                   v
                         ┌─────────────────┐
                         │ InfoNCE Loss    │
                         │ = Lower Bound   │
                         │   on MI(X; C)   │
                         └─────────────────┘
```

### Key Equivalences

**1. CPC = Predictive Coding**
- CPC predicts future observations from context
- Predictive coding predicts sensory input from generative model
- Both minimize prediction error / maximize mutual information

**2. InfoNCE = Softmax Cross-Entropy**
- InfoNCE is just cross-entropy over a similarity matrix
- Same gradient dynamics as classification
- Temperature = inverse precision in Bayesian terms

**3. Contrastive Learning = Free Energy Minimization**
- Positive pairs = expected outcomes (minimize surprise)
- Negative pairs = counterfactuals (maximize discriminability)
- Learning = reducing prediction error

**4. Self-Supervised = Prediction = Intelligence**
- Yann LeCun: "Prediction is the essence of intelligence"
- CPC, CLIP, GPT all predict "next thing"
- Different modalities, same principle

### Mathematical Unity

All these methods optimize variants of:

```
L = -log p(positive | context) / sum_i p(sample_i | context)
```

This is:
- **InfoNCE** when p is parameterized as exp(f(x,c))
- **Cross-entropy** when samples are classes
- **Variational bound** when p is approximate posterior
- **Free energy** when negative log evidence

---

## Section 7: Performance Considerations

### Computational Costs

**Memory Scaling**:
- InfoNCE requires N^2 similarity computations for N samples
- For batch size 8192 (SimCLR): 67M pairwise comparisons!
- Solution: Gradient checkpointing, mixed precision

**Effective Batch Size**:
```python
# Larger batches = more negatives = tighter MI bound
# But diminishing returns after ~4096

# Gather embeddings across GPUs for larger effective batch
import torch.distributed as dist

def gather_from_all_gpus(tensor):
    """Gather tensors from all GPUs"""
    world_size = dist.get_world_size()
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors, dim=0)
```

**Temperature Tuning**:
- tau = 0.07 typical for CLIP/SimCLR
- Lower temp = sharper distribution = harder negatives
- Too low = training instability
- Learnable temperature often works best

### Implementation Optimizations

```python
def efficient_info_nce(query, key, temperature=0.07):
    """Memory-efficient InfoNCE using chunked computation"""

    batch_size = query.size(0)
    device = query.device

    # Normalize
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)

    # Compute in chunks to save memory
    chunk_size = 1024
    total_loss = 0.0

    for i in range(0, batch_size, chunk_size):
        end_i = min(i + chunk_size, batch_size)
        query_chunk = query[i:end_i]

        # Compute similarity for this chunk
        logits = query_chunk @ key.T / temperature

        # Labels for this chunk
        labels = torch.arange(i, end_i, device=device)

        # Accumulate loss
        total_loss += F.cross_entropy(logits, labels, reduction='sum')

    return total_loss / batch_size


# Hard negative mining for better gradients
def hard_negative_mining(query, key, k=10):
    """Select hardest negatives for stronger gradients"""

    with torch.no_grad():
        # Compute all similarities
        sim = query @ key.T

        # For each query, find top-k hardest negatives
        # (most similar but not positive)
        _, hard_neg_indices = sim.topk(k + 1, dim=-1)

    return hard_neg_indices[:, 1:]  # Exclude positive (index 0)
```

---

## Section 8: ARR-COC Connection - Contrastive Relevance Learning

### Relevance as Contrastive Prediction

In ARR-COC (Attention-based Relevance Reallocation), token relevance can be learned contrastively:

```python
class ContrastiveRelevanceModule(nn.Module):
    """Learn relevance scores through contrastive prediction"""

    def __init__(self, hidden_dim: int, temperature: float = 0.1):
        super().__init__()

        # Relevance predictor
        self.relevance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Context aggregator
        self.context_net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=2
        )

        self.temperature = temperature

    def compute_contrastive_relevance(
        self,
        token_features: torch.Tensor,
        query: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance using contrastive InfoNCE framework

        Intuition: Tokens that are predictive of the query should
        receive higher relevance scores

        Args:
            token_features: [batch, num_tokens, hidden_dim]
            query: [batch, hidden_dim] task/query representation
        Returns:
            relevance_scores: [batch, num_tokens] normalized scores
        """
        batch_size, num_tokens, hidden_dim = token_features.shape

        # Get context by aggregating all tokens
        context = self.context_net(
            token_features.transpose(0, 1)
        ).mean(dim=0)  # [batch, hidden_dim]

        # Compute similarity between each token and query
        # This is like asking: "How well does this token predict the query?"
        token_features_norm = F.normalize(token_features, dim=-1)
        query_norm = F.normalize(query, dim=-1)

        # [batch, num_tokens]
        similarities = torch.bmm(
            token_features_norm,
            query_norm.unsqueeze(-1)
        ).squeeze(-1) / self.temperature

        # Relevance scores via softmax (like InfoNCE attention)
        relevance_scores = F.softmax(similarities, dim=-1)

        return relevance_scores

    def contrastive_relevance_loss(
        self,
        token_features: torch.Tensor,
        important_token_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Train relevance predictor with contrastive loss

        Important tokens should be distinguishable from unimportant ones

        Args:
            token_features: [batch, num_tokens, hidden_dim]
            important_token_mask: [batch, num_tokens] boolean mask
        Returns:
            loss: contrastive loss for relevance learning
        """
        batch_size, num_tokens, hidden_dim = token_features.shape

        # Compute pairwise similarities
        features_norm = F.normalize(token_features, dim=-1)
        similarities = torch.bmm(
            features_norm,
            features_norm.transpose(1, 2)
        ) / self.temperature  # [batch, num_tokens, num_tokens]

        # Important tokens should be similar to each other (positives)
        # Important tokens should be different from unimportant (negatives)

        total_loss = 0.0
        for b in range(batch_size):
            important_idx = important_token_mask[b].nonzero(as_tuple=True)[0]
            unimportant_idx = (~important_token_mask[b]).nonzero(as_tuple=True)[0]

            if len(important_idx) == 0 or len(unimportant_idx) == 0:
                continue

            # For each important token, compute InfoNCE
            for i in important_idx:
                # Positive: other important tokens
                pos_sim = similarities[b, i, important_idx].mean()

                # Negative: unimportant tokens
                neg_sim = similarities[b, i, unimportant_idx]

                # InfoNCE-style loss
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                labels = torch.zeros(1, dtype=torch.long, device=token_features.device)

                total_loss += F.cross_entropy(logits.unsqueeze(0), labels)

        return total_loss / batch_size


# Integration with token allocation
def allocate_tokens_contrastively(
    token_features: torch.Tensor,
    query: torch.Tensor,
    budget: int,
    module: ContrastiveRelevanceModule
) -> torch.Tensor:
    """
    Allocate computation budget based on contrastive relevance

    Returns:
        selected_indices: [batch, budget] indices of selected tokens
    """
    relevance_scores = module.compute_contrastive_relevance(
        token_features, query
    )

    # Select top-k most relevant tokens
    _, selected_indices = relevance_scores.topk(budget, dim=-1)

    return selected_indices
```

### Why Contrastive Relevance Works

**Theoretical Connection**:
- Relevant tokens should be PREDICTIVE of the task/query
- This is exactly what CPC optimizes!
- InfoNCE naturally selects "informative" samples

**Practical Benefits**:
- No need for explicit relevance labels
- Learns from task structure
- Generalizes across different queries

---

## Section 9: Extensions and Variants

### Multi-Label CPC

For multiple future prediction targets:

```python
def multi_label_info_nce(predictions, targets, num_positives=5):
    """InfoNCE with multiple positive samples per query"""

    batch_size, dim = predictions.shape

    # Compute all pairwise similarities
    sim = predictions @ targets.T / 0.07

    # Create multi-hot labels
    # Assume first num_positives targets are positive for each query
    labels = torch.zeros(batch_size, batch_size, device=predictions.device)
    for i in range(batch_size):
        for j in range(num_positives):
            pos_idx = (i * num_positives + j) % batch_size
            labels[i, pos_idx] = 1.0

    # Binary cross-entropy per element
    loss = F.binary_cross_entropy_with_logits(sim, labels)

    return loss
```

### Momentum Contrast (MoCo)

Maintain queue of negatives for larger effective batch:

```python
class MoCoQueue:
    """Momentum-updated queue for contrastive learning"""

    def __init__(self, dim: int, queue_size: int = 65536):
        self.queue = F.normalize(torch.randn(queue_size, dim), dim=-1)
        self.ptr = 0
        self.queue_size = queue_size

    def enqueue(self, keys: torch.Tensor):
        """Add new keys to queue"""
        batch_size = keys.size(0)
        keys = F.normalize(keys, dim=-1)

        end_ptr = self.ptr + batch_size
        if end_ptr > self.queue_size:
            # Wrap around
            first_part = self.queue_size - self.ptr
            self.queue[self.ptr:] = keys[:first_part]
            self.queue[:end_ptr - self.queue_size] = keys[first_part:]
        else:
            self.queue[self.ptr:end_ptr] = keys

        self.ptr = end_ptr % self.queue_size

    def get_negatives(self) -> torch.Tensor:
        """Get all keys in queue as negatives"""
        return self.queue.clone().detach()
```

---

## Sources

**Original Papers:**
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) - van den Oord et al. 2018 (arXiv:1807.03748)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) - Radford et al. 2021 (arXiv:2103.00020)
- [Data-Efficient Image Recognition with Contrastive Predictive Coding](http://proceedings.mlr.press/v119/henaff20a/henaff20a.pdf) - Henaff et al. 2020 (ICML)

**Implementations:**
- [InfoNCE PyTorch Implementation](https://github.com/RElbers/info-nce-pytorch) - RElbers
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [Open CLIP](https://github.com/mlfoundations/open_clip) - ML Foundations

**Theoretical Background:**
- [Multi-label Contrastive Predictive Coding](https://papers.neurips.cc/paper_files/paper/2020/file/5cd5058bca53951ffa7801bcdf421651-Paper.pdf) - Song et al. 2020 (NeurIPS)
- [Rethinking InfoNCE: How Many Negative Samples Do You Need?](https://www.ijcai.org/proceedings/2022/0348.pdf) - Wu et al. 2022 (IJCAI)
- [Decomposed Mutual Information Estimation for Contrastive Representation Learning](http://proceedings.mlr.press/v139/sordoni21a/sordoni21a.pdf) - Sordoni et al. 2021 (ICML)

**Tutorials and Explanations:**
- [InfoNCE Explained in Details and Implementations](https://medium.com/@mlshark/infonce-explained-in-details-and-implementations-902f28199ce6) - Allen Liang, Medium (accessed 2025-11-23)
- [What Is Noise Contrastive Estimation Loss?](https://wandb.ai/self-supervised-learning/index/reports/What-Is-Noise-Contrastive-Estimation-Loss-A-Tutorial-With-Code--Vmlldzo2NzY2OTY2) - Weights & Biases
- [Variational Bounds on Mutual Information](https://m-wiesner.github.io/Variational-Bounds-on-Mutual-Information/) - Matthew Wiesner

**Additional References:**
- [f-MUTUAL INFORMATION CONTRASTIVE LEARNING](https://openreview.net/pdf/73b855bc3a618719c1138ff97e05d22ae1a89151.pdf) - OpenReview
- [Tight Mutual Information Estimation With Contrastive Fenchel-Legendre Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/file/b5cc526f12164b2144bb2e06f2e84864-Supplemental-Conference.pdf) - Guo et al. 2022 (NeurIPS)

---

## Key Takeaways

1. **CPC learns by prediction**: Predict future in latent space, not pixel space
2. **InfoNCE = classification**: (K+1)-way classification over candidates
3. **MI lower bound**: InfoNCE bounds mutual information by log(N)
4. **Temperature matters**: Controls sharpness of distribution
5. **All contrastive methods unify**: CPC, CLIP, SimCLR share the same core principle
6. **Relevance is contrastive**: What's predictive is what's relevant

**The TRAIN STATION**: Whether you're predicting future frames, matching images to text, or finding relevant tokens - you're always maximizing mutual information through contrastive discrimination. The prediction IS the representation!
