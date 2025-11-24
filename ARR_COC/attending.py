"""
ARR_COC/attending.py - Token Allocation (Salience Realization)

Maps relevance scores to token budgets. The "attention" that emerges from
relevance realization, not a fixed mechanism.

For MVP: Simple top-K selection with fixed K=200 patches.
Future: Variable budgets (64-400 tokens per patch) based on relevance.
"""

import torch
import torch.nn as nn


class TokenAllocator(nn.Module):
    """
    Allocates vision tokens based on relevance scores.

    MVP: Select top-K patches (K=200) with uniform token budgets.
    Future: Variable LOD allocation (64-400 tokens per patch).

    This is where relevance is REALIZED - converted into actual
    computational resource allocation.
    """

    def __init__(self, K: int = 200):
        super().__init__()
        self.K = K

    def forward(
        self,
        balanced_scores: torch.Tensor,  # [B, N] relevance scores
        positions: torch.Tensor  # [B, N, 2] spatial positions
    ) -> tuple:
        """
        Allocate tokens based on relevance.

        Args:
            balanced_scores: [B, N] balanced relevance scores
            positions: [B, N, 2] (y, x) positions of patches

        Returns:
            selected_indices: [B, K] indices of top-K patches
            token_budgets: [B, K] token budget for each patch (uniform for MVP)
        """
        B, N = balanced_scores.shape

        # Top-K selection
        # Returns (values, indices)
        top_scores, top_indices = torch.topk(
            balanced_scores,
            k=min(self.K, N),  # Handle case where N < K
            dim=1,
            largest=True,
            sorted=True
        )

        # For MVP: Uniform token budgets
        # Future: Map scores to budgets in range [64, 400]
        token_budgets = torch.ones_like(top_indices, dtype=torch.float32)

        return top_indices, token_budgets


# === TESTS ===

def test_allocator():
    """Test token allocator."""
    print("Testing token allocator...")

    B, N = 2, 1024
    scores = torch.rand(B, N)
    positions = torch.randint(0, 32, (B, N, 2))

    allocator = TokenAllocator(K=200)
    indices, budgets = allocator(scores, positions)

    assert indices.shape == (B, 200)
    assert budgets.shape == (B, 200)

    # Verify indices are in valid range
    assert indices.min() >= 0
    assert indices.max() < N

    # MVP: budgets should be uniform (all 1.0)
    assert (budgets == 1.0).all()

    print("✓ Token allocator test passed")


if __name__ == "__main__":
    test_allocator()
    print("\n✓ attending.py tests passed!")
