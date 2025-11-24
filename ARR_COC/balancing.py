"""
ARR_COC/balancing.py - Adaptive Tension Balancing (Opponent Processing)

Implements Vervaeke's opponent processing for navigating cognitive tensions:
- Compress ↔ Particularize (how much detail?)
- Exploit ↔ Explore (focus vs diversify?)
- Focus ↔ Diversify (concentrate vs distribute?)

Uses learned MLP to adaptively weight the three ways of knowing based on
query context and spatial positions.
"""

import torch
import torch.nn as nn


class AdaptiveTensionBalancer(nn.Module):
    """
    Balances three ways of knowing via learned opponent processing.

    Navigates cognitive tensions dynamically based on query semantics.
    Learns to weight information, perspectival, and participatory scores
    differently depending on what the query asks for.

    For MVP: Simple MLP that takes score summaries + query embedding
    and outputs three weights (one per way of knowing).
    """

    def __init__(self, hidden_dim: int = 128, query_dim: int = 1536):
        super().__init__()

        # Input: 3 scores (each with mean, max, std) + query embedding
        # = 3*3 + query_dim = 9 + 1536 = 1545
        input_dim = 9 + query_dim

        # Small MLP for weight prediction
        self.weight_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 weights (info, persp, partic)
            nn.Softmax(dim=-1)  # Normalize to sum to 1
        )

    def compute_score_summaries(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute summary statistics (mean, max, std) for a score map.

        Args:
            scores: [B, H, W] score map

        Returns:
            summaries: [B, 3] (mean, max, std)
        """
        B = scores.shape[0]

        # Flatten spatial dimensions
        scores_flat = scores.view(B, -1)  # [B, H*W]

        mean_val = scores_flat.mean(dim=1)  # [B]
        max_val = scores_flat.max(dim=1)[0]  # [B]
        std_val = scores_flat.std(dim=1)  # [B]

        summaries = torch.stack([mean_val, max_val, std_val], dim=1)  # [B, 3]

        return summaries

    def forward(
        self,
        info_scores: torch.Tensor,  # [B, N] flattened information scores
        persp_scores: torch.Tensor,  # [B, N] flattened perspectival scores
        partic_scores: torch.Tensor,  # [B, N] flattened participatory scores
        positions: torch.Tensor,  # [B, N, 2] (y, x) positions
        query_embeds: torch.Tensor,  # [B, query_dim] query embeddings
        image_size: tuple = (32, 32)  # (H, W)
    ) -> torch.Tensor:
        """
        Balance three ways of knowing via opponent processing.

        Args:
            info_scores: [B, N] information scores (flattened)
            persp_scores: [B, N] perspectival scores (flattened)
            partic_scores: [B, N] participatory scores (flattened)
            positions: [B, N, 2] spatial positions (NOT USED in v0.1 - see Finding #7)
            query_embeds: [B, query_dim] query embeddings from text
            image_size: (H, W) image dimensions (used only to reshape for summaries)

        Returns:
            balanced_scores: [B, N] combined relevance scores
        """
        B, N = info_scores.shape
        H, W = image_size

        # Reshape scores back to 2D for summary computation
        info_2d = info_scores.view(B, H, W)
        persp_2d = persp_scores.view(B, H, W)
        partic_2d = partic_scores.view(B, H, W)

        # Compute summaries
        # NOTE (v0.1 LIMITATION): Uses only global statistics (mean, max, std)
        # This LOSES SPATIAL INFORMATION! Two images with identical stats but
        # different spatial layouts will get identical opponent processing weights.
        #
        # Example that fails:
        #   Image A: High info uniformly scattered → mean=0.7, max=0.9, std=0.2
        #   Image B: High info in top-left corner → mean=0.7, max=0.9, std=0.2
        #   → Both get SAME weights, but should be weighted differently!
        #
        # TODO (v0.2): Add spatial awareness via one of:
        #   - Quadrant summaries: 4 regions × 3 stats = [B, 12] per score
        #   - Learned spatial features: 1×1 conv → [B, 16] learned features
        #   - Spatial histograms: 10 bins → [B, 10] distribution per score
        # See AUDIT_FINDINGS.md Finding #4 for full analysis.
        info_summary = self.compute_score_summaries(info_2d)  # [B, 3]
        persp_summary = self.compute_score_summaries(persp_2d)  # [B, 3]
        partic_summary = self.compute_score_summaries(partic_2d)  # [B, 3]

        # Concatenate all summaries
        summaries = torch.cat([info_summary, persp_summary, partic_summary], dim=1)  # [B, 9]

        # Concatenate with query embeddings (NOW USES REAL QUERY!)
        balancer_input = torch.cat([summaries, query_embeds], dim=1)  # [B, 1545]

        # Predict weights for each way of knowing
        weights = self.weight_predictor(balancer_input)  # [B, 3]

        # Extract individual weights
        w_info = weights[:, 0:1]  # [B, 1]
        w_persp = weights[:, 1:2]  # [B, 1]
        w_partic = weights[:, 2:3]  # [B, 1]

        # Weighted combination of scores
        balanced_scores = (
            w_info * info_scores +
            w_persp * persp_scores +
            w_partic * partic_scores
        )  # [B, N]

        # Normalize to [0, 1]
        balanced_scores = (balanced_scores - balanced_scores.min()) / (
            balanced_scores.max() - balanced_scores.min() + 1e-8
        )

        return balanced_scores

    def get_tension_values(self) -> dict:
        """
        Get current tension values for logging.

        Returns dict with tension state (for MVP, just returns placeholder).
        In full implementation, would return learned opponent processing state.
        """
        return {
            "compress_vs_particularize": 0.5,
            "exploit_vs_explore": 0.5,
            "focus_vs_diversify": 0.5
        }


# === TESTS ===

def test_balancer():
    """Test adaptive tension balancer."""
    print("Testing adaptive tension balancer...")

    B, N = 2, 1024  # 2 images, 32x32 = 1024 patches
    info = torch.rand(B, N)
    persp = torch.rand(B, N)
    partic = torch.rand(B, N)
    positions = torch.randint(0, 32, (B, N, 2))
    query_embeds = torch.rand(B, 1536)  # Real query embeddings

    balancer = AdaptiveTensionBalancer(hidden_dim=128, query_dim=1536)
    balanced = balancer(info, persp, partic, positions, query_embeds, image_size=(32, 32))

    assert balanced.shape == (B, N)
    assert balanced.min() >= 0.0 and balanced.max() <= 1.0

    # Test tension values
    tensions = balancer.get_tension_values()
    assert isinstance(tensions, dict)

    print("✓ Adaptive tension balancer test passed")


if __name__ == "__main__":
    test_balancer()
    print("\n✓ balancing.py tests passed!")
