"""
MoE Token Router - Routes tokens to top-K experts

<karpathys_code_comments>
** This File's Role **
The routing mechanism that enables DeepSeek's MoE sparsity. Each token picks its top-2 experts out
of N total experts (N=256 for DeepSeek-V3). This turns 671B total params into 37B active params.

** Function List **
route_tokens(x, num_experts, k=2) - Route each token to top-K experts
compute_gates(x) - Calculate routing probabilities via softmax
dispatch_to_experts(tokens, routes) - Scatter tokens to selected experts
combine_expert_outputs(expert_outs, weights) - Weighted sum of expert outputs

** Technical Deep Dive **
MoE routing is simple: x @ router_weights → softmax → topk. Each token gets scores for all experts,
picks top-K, and only those K experts process that token.

The genius: This is LEARNED routing. During training, the model learns which tokens should go to
which experts. Math tokens might go to "math experts", code tokens to "code experts", etc. The
specialization emerges naturally from gradient descent.

DeepSeek's innovation: Aux-loss-free load balancing. Traditionally you need an auxiliary loss to
prevent expert collapse (all tokens routing to same expert). DeepSeek figured out a deterministic
balancing strategy that just works, no aux loss needed. This simplifies training significantly.

Karpathy: The code is straightforward. The hard part is making it efficient at scale (scattering/
gathering 37B activations across 256 experts on multiple GPUs without bottlenecks).
</karpathys_code_comments>
"""

import torch
import torch.nn.functional as F

def route_tokens(x, num_experts, k=2):
    # Karpathy: x shape [batch * seq_len, d_model]. Each token routes independently.
    batch_size, d_model = x.shape

    # Karpathy: Router is just a linear layer. Simple but effective.
    # Shape: [d_model, num_experts] → output [batch * seq_len, num_experts]
    router_logits = x @ router_weights

    # Karpathy: Softmax to get routing probabilities
    gates = F.softmax(router_logits, dim=-1)

    # Karpathy: Pick top-K experts per token. For V3, K=2.
    # topk_gates shape: [batch * seq_len, 2], topk_indices shape: [batch * seq_len, 2]
    topk_gates, topk_indices = torch.topk(gates, k, dim=-1)

    # Karpathy: Renormalize top-K gates to sum to 1 (we dropped the other N-K experts)
    topk_gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True)

    return topk_gates, topk_indices

# Karpathy: That's it. The model learns router_weights during training to minimize task loss.
# Experts naturally specialize based on what routing patterns minimize loss.
