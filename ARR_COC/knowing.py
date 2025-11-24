"""
ARR_COC/knowing.py - Three Ways of Knowing (Vervaekean Framework)

Implements Vervaeke's three dimensions of relevance realization:
1. Propositional (knowing THAT) - Information content via entropy
2. Perspectival (knowing WHAT IT'S LIKE) - Salience via edge magnitude
3. Participatory (knowing BY BEING) - Query-content coupling via learned projection

The fourth way (Procedural - knowing HOW) is implemented via the adapter
which learns compression skills during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def information_score(textures: torch.Tensor) -> torch.Tensor:
    """
    Propositional knowing: Information content via Shannon entropy.

    Measures: How much information is present in the texture channels?
    Higher entropy = more unpredictable = more information.

    Args:
        textures: [B, 13, H, W] texture array

    Returns:
        scores: [B, H, W] information scores in [0, 1]
    """
    # Softmax normalization over channel dimension
    probs = F.softmax(textures, dim=1)

    # Entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

    # Normalize to [0, 1]
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    return entropy


def perspectival_score(textures: torch.Tensor) -> torch.Tensor:
    """
    Perspectival knowing: Salience landscape via edge magnitude.

    Measures: What stands out perceptually?
    High edge magnitude = salient features = perceptually important.

    Args:
        textures: [B, 13, H, W] texture array

    Returns:
        scores: [B, H, W] perspectival scores (already in [0, 1])
    """
    # Channel 7 is Sobel magnitude (saliency proxy)
    edge_magnitude = textures[:, 7, :, :]  # [B, H, W]

    return edge_magnitude


class ParticipatoryScorer(nn.Module):
    """
    Participatory knowing: Query-content coupling via learned projection.

    Measures: How relevant is this content to my query?
    Learns to project texture features to query embedding space,
    then computes cosine similarity.

    This is transjective - emerges from the RELATIONSHIP between
    query (agent) and content (arena), not objective properties alone.
    """

    def __init__(self, texture_dim: int = 13, query_dim: int = 1536):
        super().__init__()

        # Learned projection: texture → query space
        # Uses 1x1 convolutions for spatial preservation
        self.texture_proj = nn.Sequential(
            nn.Conv2d(texture_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, query_dim, kernel_size=1)
        )

    def forward(
        self,
        textures: torch.Tensor,  # [B, 13, H, W]
        query_embeds: torch.Tensor  # [B, query_dim]
    ) -> torch.Tensor:
        """
        Compute participatory scores via query-texture similarity.

        Args:
            textures: [B, 13, H, W] texture array
            query_embeds: [B, query_dim] query embeddings from text encoder

        Returns:
            scores: [B, H, W] participatory relevance scores in [0, 1]
        """
        B, C, H, W = textures.shape

        # Project textures to query space
        # [B, 13, H, W] → [B, query_dim, H, W]
        texture_features = self.texture_proj(textures)

        # Normalize for cosine similarity
        texture_features = F.normalize(texture_features, dim=1)
        query_embeds_norm = F.normalize(query_embeds, dim=1)

        # Expand query to spatial dimensions
        query_grid = query_embeds_norm.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        query_grid = query_grid.expand(-1, -1, H, W)  # [B, D, H, W]

        # Compute cosine similarity (element-wise multiply + sum over channel dim)
        similarity = (texture_features * query_grid).sum(dim=1)  # [B, H, W]

        # Normalize to [0, 1]
        # Cosine is in [-1,1], map to [0,1]
        similarity = (similarity + 1.0) / 2.0

        return similarity


# === TESTS ===

def test_information_score():
    """Test propositional knowing."""
    print("Testing information score...")

    B, H, W = 2, 32, 32
    textures = torch.rand(B, 13, H, W)

    scores = information_score(textures)

    assert scores.shape == (B, H, W)
    assert scores.min() >= 0.0 and scores.max() <= 1.0

    print("✓ Information score test passed")


def test_perspectival_score():
    """Test perspectival knowing."""
    print("Testing perspectival score...")

    B, H, W = 2, 32, 32
    textures = torch.rand(B, 13, H, W)

    scores = perspectival_score(textures)

    assert scores.shape == (B, H, W)

    print("✓ Perspectival score test passed")


def test_participatory_scorer():
    """Test participatory knowing."""
    print("Testing participatory scorer...")

    B, H, W = 2, 32, 32
    textures = torch.rand(B, 13, H, W)
    query_embeds = torch.randn(B, 1536)

    scorer = ParticipatoryScorer(texture_dim=13, query_dim=1536)
    scores = scorer(textures, query_embeds)

    assert scores.shape == (B, H, W)
    assert scores.min() >= 0.0 and scores.max() <= 1.0

    print("✓ Participatory scorer test passed")


if __name__ == "__main__":
    test_information_score()
    test_perspectival_score()
    test_participatory_scorer()
    print("\n✓ All knowing.py tests passed!")
