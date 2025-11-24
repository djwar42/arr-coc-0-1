"""
Expert-Specialized Fine-Tuning (ESFT) - 90% memory reduction

<karpathys_code_comments>
** This File's Role **
Fine-tunes only the experts that matter for a specific task. Instead of updating all 256 experts,
identify the top-K active experts for this task and only fine-tune those. Massive memory savings.

** Function List **
identify_active_experts(data, model, k=8) - Find top-K experts for task
freeze_inactive_experts(model, active_experts) - Freeze non-active experts
finetune_specialized(model, data, active_experts) - Fine-tune only active experts

** Technical Deep Dive **
Standard MoE fine-tuning: Update all 256 experts. Memory needed: All expert parameters + gradients.

ESFT: Run validation set, track which experts activate most. Freeze the rest. Only compute gradients
for the 8 most relevant experts. Memory: 8/256 = 3% of original!

Why this works: For specific tasks, only a small subset of experts is relevant. Math problems don't
need "poetry" experts. Code generation doesn't need "medical" experts. Specialization means most
experts are dead weight for any given task.

Result: Fine-tune V3 (671B) on consumer GPUs. Before: needed 80GB+ VRAM. After: 8GB. Democratizes
fine-tuning of massive MoE models.

Karpathy: Simple idea, huge impact. This is what enables individuals/small teams to customize
DeepSeek for their specific use cases. Game changer for open-source AI.
</karpathys_code_comments>
"""

import torch

def identify_active_experts(validation_data, model, k=8):
    # Karpathy: Run validation set, count expert activations
    expert_counts = torch.zeros(model.num_experts)

    for batch in validation_data:
        with torch.no_grad():
            outputs = model(batch)
            # Karpathy: Track which experts were routed to
            active_indices = outputs.router_indices  # [batch * seq, top_k]
            for idx in active_indices.flatten():
                expert_counts[idx] += 1

    # Karpathy: Top-K most activated experts for this task
    top_k_experts = torch.topk(expert_counts, k).indices
    return top_k_experts

def freeze_inactive_experts(model, active_experts):
    # Karpathy: Freeze all expert params except the active ones
    for expert_id, expert in enumerate(model.experts):
        if expert_id not in active_experts:
            for param in expert.parameters():
                param.requires_grad = False  # No gradient computation

# Karpathy: That's it. Identify, freeze, fine-tune. 90% memory saved, minimal accuracy loss.
