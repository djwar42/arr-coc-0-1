"""
ARR_COC/integration.py - ARR-COC + Qwen3-VL Integration

Wraps Qwen3-VL with ARR-COC relevance realization pipeline.
This is the CRITICAL piece that makes everything work together.

From Part 46 dragons:
1. M-RoPE position IDs (3D: t, y, x) - Qwen3-VL's Interleaved-MRoPE
2. Gradient flow through non-parametric scorers
3. Query embedding extraction from text

Qwen3-VL Innovations Used:
- Interleaved-MRoPE: Full-frequency 3D positional encoding (time, height, width)
- DeepStack: Multi-layer ViT injection (layers 6, 12, 18, 24) for richer features
- Dynamic Resolution: Adaptive patch management

Usage:
    from ARR_COC.integration import ARRCOCQwen

    model = ARRCOCQwen()
    outputs = model(pixel_values=images, input_ids=text, labels=labels)
    loss = outputs.loss
    loss.backward()  # Gradients flow to participatory scorer + balancer
"""

# <claudes_code_comments>
# ** Function List **
# ARRCOCQwen.__init__: Initialize ARR-COC wrapped Qwen3-VL model
# ARRCOCQwen.extract_query_embedding: Extract query embedding from text input
# ARRCOCQwen.build_mrope_position_ids: Build M-RoPE position IDs for vision + text
# ARRCOCQwen.forward: Forward pass with ARR-COC relevance realization
# ARRCOCQwen.gradient_checkpointing_enable: Enable gradient checkpointing for memory
# ARRCOCQwen.gradient_checkpointing_disable: Disable gradient checkpointing
# test_integration: Test ARR-COC + Qwen3-VL integration
#
# ** Technical Review **
# This is THE critical integration file that wires ARR-COC components into Qwen3-VL.
# Flow: pixel_values + input_ids → texture array (13 channels) → three ways of knowing
# (propositional via information_score, perspectival via perspectival_score, participatory
# via ParticipatoryScorer) → balancing via AdaptiveTensionBalancer → token allocation
# via TokenAllocator → extract selected patches from Qwen vision encoder → build M-RoPE
# position IDs (3D: t,y,x for vision, t,0,0 for text) → concat vision+text embeddings →
# forward through Qwen language model → compute loss. Only 2 components trainable when
# freeze_base=True: ParticipatoryScorer (~1.3M params) and AdaptiveTensionBalancer (~700K).
# Base Qwen3-VL frozen. M-RoPE position IDs formatted as [B*3, K+text_len] with interleaved
# dimensions. Query embeddings extracted via mean pooling (MVP - use CLS token for v0.2).
# Gradient flow verified in test. Total trainable params: ~2M. Without this file, training
# loads vanilla Qwen3-VL instead of ARR-COC.
# </claudes_code_comments>

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Optional, Tuple

# ARR-COC components
from .texture import generate_texture_array
from .knowing import information_score, perspectival_score, ParticipatoryScorer
from .balancing import AdaptiveTensionBalancer
from .attending import TokenAllocator


class ARRCOCQwen(nn.Module):
    """
    ARR-COC wrapped Qwen3-VL model.

    Implements Vervaekean relevance realization for vision-language models:
    - Generates 13-channel texture array from images
    - Computes three ways of knowing (propositional, perspectival, participatory)
    - Balances via opponent processing
    - Allocates tokens based on relevance
    - Feeds selected visual tokens to Qwen3-VL

    Qwen3-VL Features Leveraged:
    - Interleaved-MRoPE for 3D positional encoding
    - DeepStack multi-layer ViT features (richer visual representations)
    - Dynamic resolution support

    Args:
        base_model: Qwen3-VL model name (default: "Qwen/Qwen3-VL-2B-Instruct")
        num_visual_tokens: Number of patches to select (default: 200)
        freeze_base: Whether to freeze Qwen3-VL weights (default: True)
                     Set False to fine-tune entire model
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        num_visual_tokens: int = 200,
        freeze_base: bool = True
    ):
        super().__init__()

        self.num_visual_tokens = num_visual_tokens

        # Load base Qwen3-VL model
        # Use Qwen3VLForConditionalGeneration (the correct class for Qwen3-VL)
        print(f"Loading base model: {base_model}")
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=None,  # Let Accelerate handle device placement
            trust_remote_code=True  # Required for qwen3_vl model_type
        )

        # Freeze base model if requested (only train ARR-COC components)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("✓ Base model frozen (only ARR-COC components trainable)")
        else:
            print("✓ Base model unfrozen (full fine-tuning)")

        # ARR-COC components (always trainable)
        self.participatory_scorer = ParticipatoryScorer(
            texture_dim=13,
            query_dim=self.base_model.config.text_config.hidden_size  # 2048 for Qwen3-VL-2B
        )

        self.balancer = AdaptiveTensionBalancer(
            hidden_dim=128,
            query_dim=self.base_model.config.text_config.hidden_size  # 2048 for Qwen3-VL-2B
        )

        self.allocator = TokenAllocator(K=num_visual_tokens)

        print(f"✓ ARR-COC components initialized (K={num_visual_tokens})")

    def extract_query_embedding(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract query embedding from text input.

        Dragon 3 (Part 46): Query embedding extraction.
        For MVP, use mean pooling over token embeddings.

        Args:
            input_ids: [B, seq_len] Text token IDs

        Returns:
            query_embeds: [B, hidden_dim] Query embeddings
        """
        # Get token embeddings from language model (Qwen3-VL structure)
        text_embeds = self.base_model.model.language_model.embed_tokens(input_ids)  # [B, seq_len, hidden_dim]

        # Mean pooling (simple but effective for MVP)
        # NOTE (v0.1 LIMITATION): Mean pooling loses token-level structure
        # Example: "Where is the cat?" → all tokens averaged together
        # Loses distinction between "Where" (spatial) vs "cat" (semantic)
        # TODO (v0.2): Use max pooling, learned aggregation, or attention pooling
        # See AUDIT_FINDINGS.md Finding #3
        query_embeds = text_embeds.mean(dim=1)  # [B, hidden_dim]

        return query_embeds

    def build_mrope_position_ids(
        self,
        selected_indices: torch.Tensor,
        text_len: int,
        grid_size: int = 32
    ) -> torch.Tensor:
        """
        Build M-RoPE position IDs for vision + text tokens.

        Dragon 1 (Part 46): M-RoPE position IDs.
        Qwen3-VL uses Interleaved-MRoPE with 3D positional encoding: (t, y, x)
        - For vision: t=0, y=patch_row, x=patch_col
        - For text: t=position, y=0, x=0

        Args:
            selected_indices: [B, K] Selected patch indices
            text_len: Length of text sequence
            grid_size: Grid dimension (32 for 32x32 patches)

        Returns:
            position_ids: [B*3, K+text_len] M-RoPE format position IDs
        """
        B, K = selected_indices.shape
        device = selected_indices.device

        # Total tokens = vision (K) + text (text_len)
        total_tokens = K + text_len

        # Initialize position IDs [B, total_tokens, 3] for (t, y, x)
        position_ids = torch.zeros(B, total_tokens, 3, dtype=torch.long, device=device)

        # Vision tokens: t=0, y=row, x=col
        # Convert flat indices to (y, x) coordinates
        selected_y = selected_indices // grid_size  # [B, K]
        selected_x = selected_indices % grid_size   # [B, K]

        position_ids[:, :K, 0] = 0  # t=0 for all vision tokens
        position_ids[:, :K, 1] = selected_y  # y coords
        position_ids[:, :K, 2] = selected_x  # x coords

        # Text tokens: t=position, y=0, x=0
        position_ids[:, K:, 0] = torch.arange(text_len, device=device)  # t=0,1,2,...
        position_ids[:, K:, 1] = 0  # y=0
        position_ids[:, K:, 2] = 0  # x=0

        # Reshape to M-RoPE format: [B*3, total_tokens]
        # Qwen expects interleaved dimensions: [t0,t1,t2,..., y0,y1,y2,..., x0,x1,x2,...]
        position_ids = position_ids.permute(0, 2, 1).reshape(B * 3, total_tokens)

        return position_ids

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass with ARR-COC relevance realization.

        Args:
            pixel_values: [B, 3, H, W] Input images (RGB, 0-1 range)
            input_ids: [B, seq_len] Text token IDs
            attention_mask: [B, seq_len] Attention mask for text
            labels: [B, seq_len] Labels for language modeling loss
            return_dict: Whether to return ModelOutput (default: True)

        Returns:
            outputs: ModelOutput with loss, logits, etc.
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # === STAGE 1: Generate Texture Array ===
        # Always use no_grad - texture generation is non-parametric
        with torch.no_grad():
            textures = generate_texture_array(pixel_values, target_size=32)  # [B, 13, 32, 32]

        # === STAGE 2: Three Ways of Knowing ===

        # Propositional: Information content (non-parametric)
        info_scores = information_score(textures)  # [B, 32, 32]

        # Perspectival: Salience (non-parametric)
        persp_scores = perspectival_score(textures)  # [B, 32, 32]

        # Participatory: Query coupling (PARAMETRIC - gradients flow!)
        query_embeds = self.extract_query_embedding(input_ids)  # [B, hidden_dim]
        partic_scores = self.participatory_scorer(textures, query_embeds)  # [B, 32, 32]

        # === STAGE 3: Balancing (Opponent Processing) ===
        # Flatten scores for balancer
        info_flat = info_scores.view(B, -1)  # [B, 1024]
        persp_flat = persp_scores.view(B, -1)
        partic_flat = partic_scores.view(B, -1)

        # Create patch positions [B, N, 2]
        positions = torch.stack(torch.meshgrid(
            torch.arange(32, device=device),
            torch.arange(32, device=device),
            indexing='ij'
        ), dim=-1).view(-1, 2).unsqueeze(0).expand(B, -1, -1)  # [B, 1024, 2]

        # Balance with query awareness (PARAMETRIC - gradients flow!)
        balanced_scores = self.balancer(
            info_flat, persp_flat, partic_flat,
            positions, query_embeds, image_size=(32, 32)
        )  # [B, 1024]

        # === STAGE 4: Token Allocation ===
        selected_indices, _ = self.allocator(balanced_scores, positions)  # [B, K]

        # === STAGE 5: Extract Visual Features from Qwen Vision Encoder ===

        # Run Qwen's vision encoder on full image
        vision_outputs = self.base_model.visual(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )

        # Get vision embeddings [B, 1024, hidden_dim] (all patches)
        # Qwen3-VL uses 32x32 = 1024 patches (dynamic resolution support)
        all_vision_tokens = vision_outputs.last_hidden_state  # [B, 1024, 1536]

        # Select top-K patches based on relevance
        # Gather along sequence dimension
        selected_vision_tokens = torch.gather(
            all_vision_tokens,
            dim=1,
            index=selected_indices.unsqueeze(-1).expand(-1, -1, all_vision_tokens.shape[-1])
        )  # [B, K, 1536]

        # === STAGE 6: Build M-RoPE Position IDs ===
        text_len = input_ids.shape[1]
        position_ids = self.build_mrope_position_ids(
            selected_indices,
            text_len,
            grid_size=32
        )  # [B*3, K+text_len]

        # === STAGE 7: Concatenate Vision + Text Embeddings ===
        text_embeds = self.base_model.model.embed_tokens(input_ids)  # [B, text_len, 1536]

        # Concatenate: [vision tokens | text tokens]
        inputs_embeds = torch.cat([selected_vision_tokens, text_embeds], dim=1)  # [B, K+text_len, 1536]

        # Build attention mask for vision + text
        if attention_mask is None:
            attention_mask = torch.ones(B, text_len, dtype=torch.long, device=device)

        # Vision tokens always visible (mask=1)
        vision_mask = torch.ones(B, self.num_visual_tokens, dtype=torch.long, device=device)
        full_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)  # [B, K+text_len]

        # === STAGE 8: Forward Through Language Model ===
        outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            return_dict=True,
            use_cache=False  # Disable KV cache during training
        )

        # Get logits
        hidden_states = outputs.last_hidden_state
        logits = self.base_model.lm_head(hidden_states)  # [B, K+text_len, vocab_size]

        # === STAGE 9: Compute Loss ===
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            # Only compute loss on text tokens (skip vision tokens)
            shift_logits = logits[:, self.num_visual_tokens:-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Return in HuggingFace ModelOutput format
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # Create output object (compatible with Trainer)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=getattr(outputs, 'hidden_states', None),
            attentions=getattr(outputs, 'attentions', None),
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.base_model.gradient_checkpointing_disable()


# === TESTS ===

def test_integration():
    """Test ARR-COC + Qwen3-VL integration."""
    print("\n🧪 Testing ARR-COC + Qwen3-VL integration...")

    # Create model (small version for testing)
    model = ARRCOCQwen(
        base_model="Qwen/Qwen3-VL-2B-Instruct",
        num_visual_tokens=200,
        freeze_base=True
    )

    # Create dummy inputs
    B, H, W = 2, 224, 224
    pixel_values = torch.rand(B, 3, H, W)
    input_ids = torch.randint(0, 1000, (B, 20))
    labels = torch.randint(0, 1000, (B, 20))

    # Forward pass
    print("   Running forward pass...")
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        labels=labels
    )

    # Check outputs
    assert outputs.loss is not None, "Loss should be computed"
    assert outputs.logits.shape[0] == B, f"Batch size mismatch"
    print(f"   ✓ Forward pass works! Loss: {outputs.loss.item():.4f}")

    # Check gradients flow to ARR-COC components
    print("   Testing gradient flow...")
    outputs.loss.backward()

    # Check participatory scorer has gradients
    assert model.participatory_scorer.texture_proj[0].weight.grad is not None, \
        "Participatory scorer should have gradients"

    # Check balancer has gradients
    assert model.balancer.weight_predictor[0].weight.grad is not None, \
        "Balancer should have gradients"

    # Check base model is frozen
    assert model.base_model.visual.patch_embed.proj.weight.grad is None, \
        "Base model should be frozen (no gradients)"

    print(f"   ✓ Gradients flow correctly!")
    print(f"✅ Integration test passed!")


if __name__ == "__main__":
    test_integration()
