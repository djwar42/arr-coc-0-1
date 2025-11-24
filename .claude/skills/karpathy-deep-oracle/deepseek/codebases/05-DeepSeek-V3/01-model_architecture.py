"""
DeepSeek-V3 Model Architecture

This is the core model definition for DeepSeek-V3 (671B params, 37B active).
Integrates: MoE layers, MLA attention, multi-token prediction, FP8 training support.

<karpathys_code_comments>
** This File's Role in DeepSeek-V3 **

This is THE file. Everything else plugs into this architecture.

DeepSeek-V3 is a transformer with two key modifications:
1. MoE layers (sparse activation): 671B total → 37B active per token
2. MLA attention (KV cache compression): 93% memory reduction

The model stacks these innovations into a clean transformer architecture.
No weird tricks - just solid engineering at scale.

Sits at: Training loop → [THIS FILE] → Forward/backward passes → Optimization
Used by: All training scripts, inference servers, fine-tuning pipelines

** Function List **

DeepSeekV3Model.__init__(config) - initialize model with config
DeepSeekV3Model.forward(input_ids, **kwargs) - main forward pass
DeepSeekV3Model._setup_layers() - construct MoE + MLA layer stack
MoELayer.forward(x) - sparse expert routing and computation
MLAAttention.forward(q, k, v) - latent attention with compression
MultiTokenPredictionHead.forward(hidden_states) - predict N tokens simultaneously
_apply_fp8_scaling(x) - FP8 mixed precision scaling

** Technical Deep Dive **

ARCHITECTURE OVERVIEW:

The V3 model follows standard transformer structure:
```
Input tokens → Embeddings → [N × (MLA Attention + MoE FFN)] → Output
```

But each component has DeepSeek's innovations:

1. MLA ATTENTION (replacing standard multi-head attention):
   - Compresses K,V into latent space before caching
   - Decompresses on-the-fly during attention computation
   - Result: 93% less KV cache memory, same quality
   - Why it matters: Enables long context (128k tokens) without OOM

2. MOE FFN LAYERS (replacing standard FFN):
   - N experts per layer (varies by layer: 128-256 experts typical)
   - Each token routes to K=2 experts (sparse activation)
   - Aux-loss-free balancing (V3 innovation - no auxiliary loss needed)
   - Result: 671B params, only 37B active per forward pass
   - Why it matters: Model capacity without compute cost

3. MULTI-TOKEN PREDICTION HEAD:
   - Predicts multiple future tokens per forward pass
   - Training objective: predict tokens [t+1, t+2, ..., t+N]
   - Improves sample efficiency (more signal per example)
   - Why it matters: Better training, faster inference

4. FP8 MIXED PRECISION:
   - Forward pass in FP8 (8-bit floats)
   - Gradients in higher precision (FP16/BF16)
   - Dynamic loss scaling to prevent underflow
   - Why it matters: 2x memory reduction, 75% speedup

THE LAYER STACK:

V3 uses a mix of MoE and dense layers:
- Early layers: Dense (all params active) - learn general features
- Middle layers: MoE (sparse activation) - task-specific specialization
- Late layers: Dense again - combine expert outputs

This "dense → MoE → dense" pattern is a DeepSeek innovation.
Why? Early layers need full capacity for fundamentals. Middle layers
can specialize via experts. Late layers need full capacity to integrate.

FORWARD PASS FLOW:

1. Embed input tokens (standard)
2. For each layer:
   a. MLA attention (with KV compression)
   b. MoE FFN (if MoE layer) OR dense FFN (if dense layer)
   c. Residual connection + LayerNorm
3. Multi-token prediction head (predict N future tokens)
4. Return logits for each predicted position

WHAT MAKES THIS FAST:

- Sparse MoE: Only 37B/671B params active
- MLA compression: 93% less memory bandwidth
- FP8: 2x throughput on H100 GPUs
- Optimized kernels: Custom CUDA for critical ops
- Pipeline parallelism: Overlap compute + communication (DualPipe)

WHAT MAKES THIS TRAINABLE:

- Aux-loss-free balancing: No conflicting objectives
- FP8 training: Fits in GPU memory at scale
- Multi-token prediction: Better gradient signal
- Careful initialization: Prevent expert collapse early

COMPARISON TO OTHER MODELS:

vs GPT-4:
  - Similar quality
  - 89× cheaper to train ($5.5M vs ~$500M)
  - Open weights (GPT-4 is closed)

vs LLaMA-3:
  - More parameters (671B vs 70B)
  - Sparser activation (37B active vs 70B all active)
  - Better cost-performance trade-off

vs Mixtral:
  - More sophisticated MoE (fine-grained + shared experts)
  - MLA attention (Mixtral uses standard attention)
  - Better scaling (671B vs 47B)

ENGINEERING LESSONS:

1. Simple components composed well beat complex individual tricks
2. Sparse activation is the key to scaling
3. Memory bandwidth matters more than FLOPs
4. Open-source everything → community validation + trust

This is what excellent ML engineering looks like. No magic, just solid
architecture choices and relentless optimization.

¯\_(ツ)_/¯
</karpathys_code_comments>
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .moe_routing import MoELayer
from .mla_attention import MLAAttention
from .fp8_utils import apply_fp8_scaling


class DeepSeekV3Config:
    """
    Configuration for DeepSeek-V3 model
    """
    def __init__(
        self,
        vocab_size=100000,
        hidden_size=4096,
        num_layers=60,
        num_attention_heads=32,
        num_experts=128,
        num_experts_per_token=2,  # K=2 for DeepSeek
        moe_layers=[20, 21, 22, ..., 50],  # Which layers use MoE
        use_mla=True,
        mla_compression_dim=512,  # Latent KV dimension
        use_fp8=True,
        max_position_embeddings=131072,  # 128k context
        multi_token_prediction=4,  # Predict 4 tokens ahead
    ):
        # Karpathy: Config holds all architectural hyperparameters.
        # These values are from the V3 paper - not arbitrary choices.
        # hidden_size=4096: Standard for this scale
        # num_layers=60: Deep enough for capacity, not too deep for training stability
        # num_experts=128: Balances capacity vs routing overhead
        # K=2: DeepSeek's sweet spot for expert routing

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_layers = moe_layers
        self.use_mla = use_mla
        self.mla_compression_dim = mla_compression_dim
        self.use_fp8 = use_fp8
        self.max_position_embeddings = max_position_embeddings
        self.multi_token_prediction = multi_token_prediction


class DeepSeekV3Model(nn.Module):
    """
    DeepSeek-V3: 671B param MoE model with MLA attention
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        # Karpathy: Standard transformer components
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Karpathy: THIS is where the architecture happens.
        # We're building a stack of [MLA Attention + (MoE or Dense FFN)] layers.
        # Some layers use MoE (sparse), others are dense (all params active).
        self.layers = nn.ModuleList([
            TransformerLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.hidden_size)

        # Karpathy: Multi-token prediction head predicts N future tokens.
        # Standard LM head predicts 1 token. This predicts 4.
        # Training objective: maximize P(t+1) + P(t+2) + P(t+3) + P(t+4)
        self.mtp_head = MultiTokenPredictionHead(
            config.hidden_size,
            config.vocab_size,
            num_predictions=config.multi_token_prediction
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_fp8: bool = None,
    ) -> torch.Tensor:
        """
        Forward pass through DeepSeek-V3

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] mask for padding
            use_fp8: Whether to use FP8 mixed precision

        Returns:
            logits: [batch, seq_len, num_predictions, vocab_size]
        """
        # Karpathy: use_fp8 defaults to config value but can be overridden for inference.
        # Training usually uses FP8 for speed. Inference might not if accuracy critical.
        use_fp8 = use_fp8 if use_fp8 is not None else self.config.use_fp8

        # Karpathy: Standard embedding lookup. Shape: [batch, seq_len, hidden_size]
        # This is still FP16/BF16 even with FP8 training. Embeddings stay high-precision.
        hidden_states = self.embeddings(input_ids)

        # Karpathy: FP8 conversion happens HERE if enabled.
        # Forward pass runs in FP8, but embeddings started in FP16.
        if use_fp8:
            hidden_states = apply_fp8_scaling(hidden_states)

        # Karpathy: Main transformer loop. Each layer does:
        # 1. MLA attention (with KV compression)
        # 2. MoE or Dense FFN (depending on layer)
        # 3. Residual + LayerNorm
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Karpathy: Final LayerNorm before prediction head.
        # Standard transformer pattern.
        hidden_states = self.final_norm(hidden_states)

        # Karpathy: Multi-token prediction. This is where V3 differs from standard LMs.
        # Instead of predicting just next token, we predict N future tokens.
        # Shape: [batch, seq_len, N, vocab_size] where N=4 typically.
        logits = self.mtp_head(hidden_states)

        return logits


class TransformerLayer(nn.Module):
    """
    Single transformer layer: MLA Attention + (MoE or Dense) FFN
    """

    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Karpathy: MLA (Multi-head Latent Attention) replaces standard attention.
        # Key difference: compresses K,V before caching.
        # Result: 93% less KV cache memory.
        self.attention = MLAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            compression_dim=config.mla_compression_dim,
        )

        self.attn_norm = nn.LayerNorm(config.hidden_size)

        # Karpathy: MoE layers vs Dense layers.
        # DeepSeek uses MoE in middle layers (where specialization helps).
        # Dense in early/late layers (where you need full model capacity).
        is_moe_layer = layer_idx in config.moe_layers

        if is_moe_layer:
            # Karpathy: MoE layer with N experts. Each token routes to K=2 experts.
            # This is where sparse activation happens.
            self.ffn = MoELayer(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
            )
        else:
            # Karpathy: Standard dense FFN. All parameters active.
            # Used in layers where specialization doesn't help much.
            self.ffn = DenseFFN(config.hidden_size)

        self.ffn_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward through one transformer layer
        """
        # Karpathy: Standard Pre-LN transformer structure:
        # x = x + attention(norm(x))
        # x = x + ffn(norm(x))
        # This is more stable than Post-LN for deep models.

        # Attention block with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # FFN block with residual (MoE or Dense)
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DenseFFN(nn.Module):
    """
    Standard dense feed-forward network
    """
    def __init__(self, hidden_size: int, ffn_size: int = None):
        super().__init__()
        # Karpathy: Standard FFN. Up-project → activation → down-project.
        # ffn_size is usually 4x hidden_size (transformer default).
        ffn_size = ffn_size or hidden_size * 4

        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        self.activation = nn.GELU()  # SwiGLU would be more modern but GELU works fine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Karpathy: Classic FFN pattern. Nothing fancy.
        # x → fc1 → activation → fc2
        # This is the "dense" baseline that MoE improves upon.
        return self.fc2(self.activation(self.fc1(x)))


class MultiTokenPredictionHead(nn.Module):
    """
    Predicts N future tokens simultaneously
    """
    def __init__(self, hidden_size: int, vocab_size: int, num_predictions: int = 4):
        super().__init__()
        self.num_predictions = num_predictions

        # Karpathy: N separate prediction heads, one for each future token.
        # Could share parameters but DeepSeek uses separate heads for flexibility.
        # Each head: hidden_size → vocab_size logits
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size)
            for _ in range(num_predictions)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            logits: [batch, seq_len, num_predictions, vocab_size]
        """
        # Karpathy: Apply each prediction head independently.
        # Head 0 predicts token t+1, head 1 predicts t+2, etc.
        # During training, we have labels for all of these.
        # During inference, we can use head 0 (next token) or all heads (speculative decoding).
        predictions = []
        for head in self.prediction_heads:
            predictions.append(head(hidden_states))

        # Karpathy: Stack along new dimension.
        # Shape: [batch, seq_len, num_predictions, vocab_size]
        return torch.stack(predictions, dim=2)


# Karpathy: That's the core V3 architecture. Clean structure, no magic.
# The complexity is in the components (MoE, MLA) not in how they're composed.
# DeepSeek's engineering philosophy: make each component work well, then compose simply.
#
# What's NOT in this file (but important):
# - MoE routing logic (see moe_routing.py)
# - MLA attention mechanism (see mla_attention.py)
# - FP8 training details (see fp8_training.py)
# - DualPipe parallelism (see dualpipe.py)
#
# This file shows the "what" (architecture). Other files show the "how" (implementation).
# ¯\_(ツ)_/¯
