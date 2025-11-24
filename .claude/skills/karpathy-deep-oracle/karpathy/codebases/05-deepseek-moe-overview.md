# DeepSeekMoE Architecture Overview

## Summary

**DeepSeekMoE 16B** is a Mixture-of-Experts language model with 16.4B total parameters but only ~2.8B activated per token (~40% computation of equivalent dense models). Trained on 2T English/Chinese tokens, it achieves comparable performance to DeepSeek 7B and LLaMA2 7B with significantly reduced compute.

## Key Innovations

### 1. Fine-Grained Expert Segmentation
- **64 routed experts** instead of traditional large monolithic experts
- Enables finer-grained specialization per knowledge domain
- Each token routes to **top-K=6 experts** via learned gating
- Smaller expert size allows more precise routing decisions

### 2. Shared Experts Isolation
- **Dedicated shared experts** always activated (handle common knowledge)
- **Routed experts** specialize in specific domains
- Reduces routing pressure and improves expert utilization
- Shared experts add residually to routed expert outputs

## Architecture Components

### Core MoE Classes

**MoEGate (modeling_deepseek.py:280)**
```
Purpose: Routes tokens to top-K experts with load balancing
Key features:
- Softmax-based routing: scores = softmax(linear(hidden_states))
- Top-K selection with weight normalization (sum to 1)
- Auxiliary loss for expert load balancing (prevents collapse)
- Two modes: token-level or sequence-level balancing
```

**DeepseekMoE (modeling_deepseek.py:361)**
```
Purpose: Core MoE layer combining routed and shared experts
Architecture:
- n_routed_experts (64) small MLP experts
- n_shared_experts (2) always-active experts
- Separate training/inference paths for efficiency
Flow:
  hidden_states → MoEGate (routing) → routed_experts → weighted_sum
                                   └→ shared_experts (always) → add
```

**AddAuxiliaryLoss (modeling_deepseek.py:341)**
```
Purpose: Custom autograd function for auxiliary loss gradient flow
Trick: Adds aux_loss gradient during backward pass without modifying forward
Ensures load balancing loss propagates through expert routing
```

### Training vs Inference

**Training Path (modeling_deepseek.py:381-388)**
- Repeat-interleave tokens for parallel expert computation
- All selected experts process in parallel
- Auxiliary loss computed for load balancing
- Gradient flows through routing weights

**Inference Path (modeling_deepseek.py:395-411)**
- Sort tokens by assigned expert_id
- Batch process tokens per expert sequentially
- More cache-efficient than training path
- No auxiliary loss computation

### Efficiency Optimizations

**Memory**
- RMSNorm instead of LayerNorm (simpler, numerically stable)
- Grouped-query attention (GQA) reduces KV cache
- Flash Attention 2 for O(N) memory complexity
- DeepSpeed ZeRO-3 for distributed training (offload to CPU)

**Compute**
- Only 2.8B / 16.4B params activated per forward pass
- Top-K gating (K=6) limits expert activation
- Shared experts amortize common knowledge compute
- QLoRA enables finetuning on single A100 80GB

## Finetuning Infrastructure

**finetune.py** - Supervised finetuning script with:
- LoRA/QLoRA support (4-bit/8-bit quantization)
- DeepSpeed ZeRO-2/ZeRO-3 integration
- Flash Attention 2 support
- Automatic checkpoint resumption
- Custom PEFT adapter saving

**Key Parameters**
```python
# LoRA config
lora_rank = 8-64  # Rank of low-rank decomposition
lora_alpha = 16-32  # Scaling factor
target_modules = "q,k,v,o,gate,down,up"  # All projections

# Quantization (QLoRA)
bits = 4  # 4-bit NF4 quantization
double_quant = True  # Compress quantization stats
```

## Files Structure

```
05-DeepSeek-MoE/
├── modeling_deepseek.py (1559 lines) - Core MoE architecture
│   ├── MoEGate - Routing mechanism with load balancing
│   ├── DeepseekMoE - Main MoE layer (routed + shared experts)
│   ├── DeepseekDecoderLayer - Transformer layer with MoE
│   ├── DeepseekModel - Full transformer model
│   └── DeepseekForCausalLM - Causal LM with generation
│
├── finetune/
│   ├── finetune.py (323 lines) - Training script
│   └── configs/
│       ├── ds_config_zero2_no_offload.json - ZeRO-2 config
│       └── ds_config_zero3.json - ZeRO-3 with CPU offload
│
├── README.md - Usage and model documentation
├── DeepSeekMoE.pdf - Research paper
└── requirements.txt - Dependencies
```

## Mathematical Foundations

### Auxiliary Load Balancing Loss

**Token-level balancing:**
```
ce[i] = fraction of tokens routed to expert i
Pi[i] = mean routing score for expert i
fi[i] = ce[i] * n_routed_experts
aux_loss = alpha * sum(Pi[i] * fi[i])
```

**Sequence-level balancing:**
```
ce[i] = fraction of tokens per sequence to expert i
aux_loss = alpha * mean_over_batch(sum(ce * mean_scores))
```

Goal: Encourages uniform expert utilization, prevents routing collapse

### MoE Forward Pass

```
# Routing
scores = softmax(W_gate @ hidden_states)  # [batch*seq, n_experts]
topk_weights, topk_indices = topk(scores, k=6)  # Top-6 experts
topk_weights = topk_weights / sum(topk_weights)  # Normalize

# Expert computation (training)
outputs = []
for expert_id in range(n_routed_experts):
    mask = (topk_indices == expert_id)
    outputs[expert_id] = expert[expert_id](hidden_states[mask])

# Weighted combination
routed_output = sum(topk_weights * expert_outputs)
final_output = routed_output + shared_experts(hidden_states)
```

## Performance Characteristics

**Computational Efficiency**
- 16.4B total params → 2.8B activated (17% activation rate)
- 40% FLOPs of DeepSeek 7B dense model
- Matches DeepSeek 7B performance on most benchmarks
- 39.6% FLOPs of LLaMA2 7B with comparable/better performance

**Memory Requirements**
- Inference: Single GPU with 40GB VRAM (no quantization)
- Training full precision: 8×A100 40GB with ZeRO-3
- Training QLoRA 4-bit: Single A100 80GB
- Finetuning LoRA: 8×A100 40GB with ZeRO-2

## Design Rationale

**Why Fine-Grained Experts?**
- Smaller experts → more routing choices → better specialization
- 64 experts vs typical 8-16 in other MoE models
- Top-6 routing maintains reasonable computation budget

**Why Shared Experts?**
- Handle common knowledge (frequent patterns, basic grammar)
- Reduce pressure on routing mechanism
- Improve training stability (always-active gradient path)
- 2 shared experts × larger size ≈ baseline FFN capacity

**Why Auxiliary Loss?**
- Prevent routing collapse (all tokens to few experts)
- Encourage expert diversity and specialization
- Balance expert utilization for efficient compute
- Alpha hyperparameter tunes exploration vs exploitation

## Connections to Other Work

**Related to DeepSeek-V3** (source-codebases/07-DeepSeek-V3)
- V3 extends MoE architecture to 671B params
- Uses similar fine-grained + shared expert design
- Implements Multi-head Latent Attention (MLA) for efficiency

**Related to DualPipe** (source-codebases/09-DualPipe)
- Pipeline parallelism for MoE models
- Optimizes expert-parallel + data-parallel training
- Critical for scaling to V3-scale models

**Related to FlashMLA** (source-codebases/11-FlashMLA)
- Memory-efficient attention for MoE models
- Reduces KV cache overhead
- Enables longer context windows

## Karpathy's Efficiency Analysis Connection

From `codebase/02-karpathy-on-deepseek-efficiency.md`:
> "The MoE architecture is fascinating - you get model capacity without proportional compute. The key insight is expert specialization through fine-grained routing."

DeepSeekMoE validates this through:
- 40% compute reduction vs dense models
- Maintained performance through specialization
- Shared experts handling common patterns efficiently

## Usage Examples

**Inference**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")

text = "An attention function can be described as"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
```

**Finetuning (QLoRA)**
```bash
deepspeed finetune/finetune.py \
    --model_name_or_path deepseek-ai/deepseek-moe-16b-base \
    --data_path data.parquet \
    --output_dir ./checkpoints \
    --bf16 True \
    --use_lora True \
    --bits 4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --deepspeed configs/ds_config_zero2_no_offload.json
```

## Open Questions for Deep Dive (Phase 2)

1. How does expert specialization emerge during training?
2. What patterns do shared vs routed experts learn?
3. How sensitive is performance to auxiliary loss alpha?
4. What's the optimal number of experts vs expert size?
5. How does routing change during finetuning?

## References

- Paper: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (arXiv:2401.06066)
- HuggingFace: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
- Code: modeling_deepseek.py (1559 lines)
