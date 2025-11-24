# Part 38 Addendum: HuggingFace Implementation Plan
*A practical roadmap for building ARR-COC-VIS with Qwen3-VL-2B-Instruct, including side-by-side testing interface and comprehensive evaluation metrics*

---

## Overview

**Target Model:** Qwen/Qwen3-VL-2B-Instruct (2B params, 74.8k downloads, most stable 2B variant)

**Implementation Strategy:** Start with weighted attention (Strategy 2 from Part 33), validate hypotheses, then optimize to sparse sampling (Strategy 1) if successful.

---

## Phase 1: Foundation

### 1.1 Environment Setup

```bash
# Create conda environment
conda create -n arr-coc python=3.10
conda activate arr-coc

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.57.0  # Latest with Qwen3-VL support
pip install gradio pillow numpy pandas matplotlib seaborn
pip install kornia  # GPU-accelerated CV operations
pip install opencv-python scikit-image

# Optional: Flash Attention 2 for efficiency
pip install flash-attn --no-build-isolation
```

### 1.2 Repository Structure

```
arr-coc-ovis/
├── arr_coc/
│   ├── __init__.py
│   ├── texture_array.py      # 40-channel generation
│   ├── knowing.py             # Three scorers
│   ├── balancing.py           # Contextual tension balancer
│   ├── attending.py           # Token allocator
│   ├── realizing.py           # Pipeline orchestrator
│   └── qwen_integration.py    # Qwen3-VL wrapper
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # Accuracy, efficiency, robustness
│   ├── visualize.py           # Heatmaps, attention plots
│   └── benchmark.py           # Dataset evaluation
│
├── tests/
│   ├── test_texture_array.py
│   ├── test_scorers.py
│   └── test_integration.py
│
├── app.py                     # Main Gradio interface
├── train.py                   # Training script
├── config.yaml                # Hyperparameters
└── README.md
```

### 1.3 Baseline Qwen3-VL Test

**File: `tests/test_baseline.py`**

```python
"""
Test baseline Qwen3-VL performance before ARR-COC integration.
Establish ground truth for comparison.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import time

def test_baseline():
    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    # Test image
    image = Image.open("test_images/sample.jpg")
    query = "What objects are in this image?"

    # Prepare messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query}
        ]
    }]

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Decode
    answer = processor.batch_decode(
        outputs[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0]

    # Memory usage
    memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Answer: {answer}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print(f"Memory: {memory_mb:.2f}MB")
    print(f"Num patches: {inputs['pixel_values'].shape[1]}")

    return {
        'answer': answer,
        'time_ms': elapsed * 1000,
        'memory_mb': memory_mb,
        'num_patches': inputs['pixel_values'].shape[1]
    }

if __name__ == "__main__":
    baseline = test_baseline()
```

---

## Phase 2: MVP Components

### 2.1 Simplified Texture Array (13 channels for MVP)

**File: `arr_coc/texture_array.py`**

```python
"""
Texture array generation for ARR-COC-VIS.
MVP: 13 fast channels only (skip expensive SAM/OCR/CLIP initially).
"""

import torch
import kornia
from typing import Tuple

class TextureArrayGenerator:
    """Generate multi-channel texture array from image"""

    def __init__(self, channels='mvp'):
        """
        Args:
            channels: 'mvp' (13 channels) or 'full' (40 channels)
        """
        self.channels = channels

    def generate(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate texture array from image.

        Args:
            image: [3, H, W] RGB image tensor (normalized 0-1)

        Returns:
            texture: [C, H, W] multi-channel texture
                     C=13 for MVP, C=40 for full
        """
        device = image.device
        H, W = image.shape[1:]

        if self.channels == 'mvp':
            return self._generate_mvp(image)
        else:
            return self._generate_full(image)

    def _generate_mvp(self, image: torch.Tensor) -> torch.Tensor:
        """Generate 13-channel MVP texture array (fast)"""
        device = image.device
        H, W = image.shape[1:]

        texture = torch.zeros(13, H, W, device=device, dtype=image.dtype)

        # Channels 0-2: RGB
        texture[0:3] = image

        # Channels 3-5: Position encoding
        y_coords = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, W).expand(H, W)

        # Eccentricity (distance from center)
        cy, cx = H // 2, W // 2
        eccentricity = torch.sqrt(
            ((y_coords - 0.5) ** 2 + (x_coords - 0.5) ** 2)
        )

        texture[3] = y_coords
        texture[4] = x_coords
        texture[5] = eccentricity

        # Channels 6-7: Edges (normal + inverted)
        gray = image.mean(dim=0, keepdim=True)
        edges = kornia.filters.sobel(gray.unsqueeze(0)).squeeze(0).abs().mean(dim=0)

        texture[6] = edges
        texture[7] = 1.0 - edges  # Inverted (catches low-contrast text)

        # Channels 8-9: Highpass/Lowpass
        texture[8] = kornia.filters.laplacian(gray.unsqueeze(0), kernel_size=3).squeeze()
        texture[9] = kornia.filters.gaussian_blur2d(
            gray.unsqueeze(0), (5, 5), (1.0, 1.0)
        ).squeeze()

        # Channel 10: Motion (placeholder for MVP, zeros for static images)
        texture[10] = torch.zeros_like(edges)

        # Channel 11: Saliency (simple center bias for MVP)
        texture[11] = 1.0 - 0.7 * eccentricity  # Foveal bias

        # Channel 12: Distance field (placeholder for MVP)
        texture[12] = torch.zeros_like(edges)

        return texture

    def _generate_full(self, image: torch.Tensor) -> torch.Tensor:
        """Generate full 40-channel texture array (slower, for later)"""
        # TODO: Implement after MVP validation
        # Includes: clusters (SAM), text regions (OCR), CLIP embeddings
        raise NotImplementedError("Full 40-channel array not yet implemented")
```

### 2.2 Three Ways of Knowing (Scorers)

**File: `arr_coc/knowing.py`**

```python
"""
Three Ways of Knowing: Propositional, Perspectival, Participatory
Based on John Vervaeke's relevance realization framework.
"""

import torch
import torch.nn as nn

class InformationScorer(nn.Module):
    """Propositional knowing - objective information content"""

    def __init__(self):
        super().__init__()
        # Learnable weights for combining information channels
        self.weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))

    def forward(self, texture: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Score positions by information content.

        Args:
            texture: [13, H, W] texture array
            positions: [N, 2] (y, x) coordinates

        Returns:
            scores: [N] propositional scores
        """
        N = positions.shape[0]

        # Extract features at positions
        # Channel 6: edges, 7: inverted edges, 8: highpass
        y_idx = positions[:, 0].long()
        x_idx = positions[:, 1].long()

        edges = texture[6, y_idx, x_idx]
        edges_inv = texture[7, y_idx, x_idx]
        highpass = texture[8, y_idx, x_idx]

        # Combine with learned weights
        edge_max = torch.maximum(edges, edges_inv)

        scores = (
            self.weights[0] * edge_max +
            self.weights[1] * highpass.abs() +
            self.weights[2] * edges  # Structure
        )

        return scores


class PerspectivalScorer(nn.Module):
    """Perspectival knowing - salience landscape"""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(self, texture: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Score positions by perspectival salience.

        Args:
            texture: [13, H, W] texture array
            positions: [N, 2] coordinates

        Returns:
            scores: [N] perspectival scores
        """
        y_idx = positions[:, 0].long()
        x_idx = positions[:, 1].long()

        # Channel 11: saliency, 10: motion, 5: eccentricity
        saliency = texture[11, y_idx, x_idx]
        motion = texture[10, y_idx, x_idx]
        eccentricity = texture[5, y_idx, x_idx]

        # Foveal bias (center regions weighted higher)
        foveal_weight = 1.0 - 0.5 * eccentricity

        scores = (
            self.weights[0] * saliency +
            self.weights[1] * motion
        ) * foveal_weight

        return scores


class ParticipatoryScorer(nn.Module):
    """Participatory knowing - query-content coupling"""

    def __init__(self):
        super().__init__()
        # For MVP: Use simple text-image similarity (placeholder)
        # Later: Use CLIP embeddings from channels 17-32

    def forward(self, texture: torch.Tensor, positions: torch.Tensor,
                query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Score positions by participatory coupling with query.

        Args:
            texture: [13, H, W] texture array
            positions: [N, 2] coordinates
            query_embedding: [D] encoded query (placeholder for MVP)

        Returns:
            scores: [N] participatory scores
        """
        N = positions.shape[0]

        # MVP: Return uniform scores (no CLIP yet)
        # TODO: Integrate CLIP embeddings when adding channels 17-32
        return torch.ones(N, device=texture.device) * 0.5
```

### 2.3 Contextual Tension Balancer (Part 37's innovation)

**File: `arr_coc/balancing.py`**

```python
"""
Contextual Tension Balancer - navigates opponent processes.
Key insight from Part 37: Tensions must adapt to context, not be fixed.
"""

import torch
import torch.nn as nn

class ContextualTensionBalancer(nn.Module):
    """
    Balance three scorers with adaptive tensions.
    Tensions = f(query, image, score_statistics) not constants!
    """

    def __init__(self, context_dim=128):
        super().__init__()

        # Policy network: context → tensions
        self.tension_policy = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [compress_vs_particularize, exploit_vs_explore, focus_vs_diversify]
        )

        # Combiner: (scores, tensions) → balanced_scores
        self.combiner = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def compute_context(self, info_scores, persp_scores, partic_scores,
                       query_embedding=None, image_features=None):
        """
        Extract context vector from scores and embeddings.

        Returns:
            context: [context_dim] context vector
        """
        # Score statistics
        score_stats = torch.tensor([
            info_scores.mean().item(),
            info_scores.std().item(),
            persp_scores.mean().item(),
            persp_scores.std().item(),
            partic_scores.mean().item(),
            partic_scores.std().item(),
        ], device=info_scores.device)

        # For MVP: Use score stats only
        # Later: Concatenate query_embedding + image_features
        context = torch.cat([
            score_stats,
            torch.zeros(122, device=score_stats.device)  # Pad to 128D
        ])

        return context

    def forward(self, info_scores, persp_scores, partic_scores,
                query_embedding=None, image_features=None):
        """
        Balance scores using adaptive tensions.

        Returns:
            balanced_scores: [N] combined relevance scores
            tensions: [3] adaptive tension values
        """
        # Compute context
        context = self.compute_context(
            info_scores, persp_scores, partic_scores,
            query_embedding, image_features
        )

        # Policy: context → tensions
        tension_logits = self.tension_policy(context)
        tensions = torch.sigmoid(tension_logits)  # [0, 1]

        # Tensions are now ADAPTIVE based on context!
        # Query "small text?" → compress=0.15
        # Query "describe" → compress=0.85

        # Stack scores
        scores = torch.stack([
            info_scores,
            persp_scores,
            partic_scores
        ], dim=1)  # [N, 3]

        # Combine using tensions (broadcast)
        weighted_scores = scores * tensions.unsqueeze(0)  # [N, 3]

        # MLP combiner
        balanced_scores = self.combiner(weighted_scores).squeeze(-1)  # [N]

        return balanced_scores, tensions
```

### 2.4 Token Allocator

**File: `arr_coc/attending.py`**

```python
"""
Token allocation - map relevance to budgets.
Implements the homunculus: variable cortical magnification.
"""

import torch
import torch.nn as nn

class TokenAllocator(nn.Module):
    """Allocate tokens based on relevance scores"""

    def __init__(self, num_positions=273, min_budget=64, max_budget=400,
                 total_budget=75000):
        super().__init__()
        self.num_positions = num_positions
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.total_budget = total_budget

        # Learnable allocation curve steepness
        self.steepness = nn.Parameter(torch.tensor(3.2))

    def forward(self, scores: torch.Tensor, positions: torch.Tensor):
        """
        Allocate token budgets to positions.

        Args:
            scores: [N] relevance scores
            positions: [N, 2] candidate positions

        Returns:
            selected_positions: [num_positions, 2]
            budgets: [num_positions] token counts
        """
        N = scores.shape[0]

        # Select top-k positions
        top_indices = torch.topk(scores, k=self.num_positions).indices
        selected_positions = positions[top_indices]
        selected_scores = scores[top_indices]

        # Normalize scores to [0, 1]
        score_min = selected_scores.min()
        score_max = selected_scores.max()

        if score_max > score_min:
            normalized = (selected_scores - score_min) / (score_max - score_min)
        else:
            normalized = torch.ones_like(selected_scores) * 0.5

        # Map to budgets using power curve
        # High scores → max_budget, low scores → min_budget
        budget_range = self.max_budget - self.min_budget
        raw_budgets = self.min_budget + budget_range * (normalized ** self.steepness)

        # Scale to fit total budget
        total_allocated = raw_budgets.sum()
        if total_allocated > self.total_budget:
            scale = self.total_budget / total_allocated
            budgets = raw_budgets * scale
        else:
            budgets = raw_budgets

        return selected_positions, budgets.long()
```

---

## Phase 3: Integration with Qwen3-VL

### 3.1 Qwen Integration Wrapper

**File: `arr_coc/qwen_integration.py`**

```python
"""
Integration with Qwen3-VL using weighted attention strategy.
Strategy 2 from Part 33: Process all patches, weight by relevance.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

class ARR_COC_Qwen:
    """ARR-COC-VIS integrated with Qwen3-VL"""

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct"):
        # Load Qwen3-VL
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # ARR-COC components (will be initialized separately)
        self.texture_gen = None
        self.info_scorer = None
        self.persp_scorer = None
        self.partic_scorer = None
        self.balancer = None
        self.allocator = None

    def allocate_relevance(self, image: Image.Image, query: str):
        """
        Run ARR-COC pipeline to compute relevance allocation.

        Returns:
            positions: [273, 2] selected positions
            budgets: [273] token budgets
            relevance_map: [H, W] full relevance heatmap
        """
        # Convert to tensor
        image_tensor = torch.from_numpy(
            np.array(image).transpose(2, 0, 1)
        ).float() / 255.0
        image_tensor = image_tensor.to(self.model.device)

        # Generate texture array
        texture = self.texture_gen.generate(image_tensor)

        # Sample candidate positions (grid for MVP)
        H, W = texture.shape[1:]
        stride = 16
        y_coords = torch.arange(0, H, stride, device=self.model.device)
        x_coords = torch.arange(0, W, stride, device=self.model.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        positions = torch.stack([yy.flatten(), xx.flatten()], dim=1)

        # Score positions
        info_scores = self.info_scorer(texture, positions)
        persp_scores = self.persp_scorer(texture, positions)
        partic_scores = self.partic_scorer(texture, positions, query_embedding=None)

        # Balance scores
        balanced_scores, tensions = self.balancer(
            info_scores, persp_scores, partic_scores
        )

        # Allocate tokens
        selected_positions, budgets = self.allocator(balanced_scores, positions)

        # Create full relevance map for visualization
        relevance_map = torch.zeros(H, W, device=self.model.device)
        for pos, score in zip(positions, balanced_scores):
            y, x = pos.long()
            relevance_map[y, x] = score

        return selected_positions, budgets, relevance_map, tensions

    def generate(self, image: Image.Image, query: str, use_arr_coc=True):
        """
        Generate answer using Qwen3-VL with optional ARR-COC.

        Args:
            image: PIL Image
            query: Text query
            use_arr_coc: If True, use ARR-COC weighting. If False, standard Qwen.

        Returns:
            dict with answer, stats, relevance_map (if arr_coc=True)
        """
        import time

        # ARR-COC allocation (if enabled)
        if use_arr_coc and self.allocator is not None:
            positions, budgets, relevance_map, tensions = self.allocate_relevance(image, query)
        else:
            relevance_map = None
            tensions = None

        # Standard Qwen3-VL processing
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]
        }]

        torch.cuda.synchronize()
        start_time = time.time()

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(self.model.device)

        # TODO: Inject relevance weights into attention here
        # For MVP: Just measure allocation, don't actually modify Qwen yet

        outputs = self.model.generate(**inputs, max_new_tokens=128)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Decode
        answer = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]

        # Stats
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        num_patches = inputs['pixel_values'].shape[1] if 'pixel_values' in inputs else 0

        return {
            'answer': answer,
            'time_ms': elapsed * 1000,
            'memory_mb': memory_mb,
            'num_patches': num_patches,
            'relevance_map': relevance_map,
            'tensions': tensions.tolist() if tensions is not None else None
        }
```

---

## Phase 4: Evaluation & Comparison Interface

### 4.1 Main App with Side-by-Side Comparison

**File: `app.py`**

```python
"""
ARR-COC-VIS Interactive Demo
Side-by-side comparison: Standard Qwen3-VL vs ARR-COC-enhanced

Features:
- Upload image + ask question
- See both models' answers
- View relevance allocation heatmap
- Compare efficiency metrics
- Aggregate statistics over multiple queries
"""

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime

# Import our components
from arr_coc.texture_array import TextureArrayGenerator
from arr_coc.knowing import InformationScorer, PerspectivalScorer, ParticipatoryScorer
from arr_coc.balancing import ContextualTensionBalancer
from arr_coc.attending import TokenAllocator
from arr_coc.qwen_integration import ARR_COC_Qwen

# Global state
arr_coc_model = None
session_stats = []

def initialize_models():
    """Initialize ARR-COC components and Qwen3-VL"""
    global arr_coc_model

    print("Loading Qwen3-VL-2B-Instruct...")
    arr_coc_model = ARR_COC_Qwen("Qwen/Qwen3-VL-2B-Instruct")

    # Initialize ARR-COC components
    print("Initializing ARR-COC components...")
    arr_coc_model.texture_gen = TextureArrayGenerator(channels='mvp')
    arr_coc_model.info_scorer = InformationScorer()
    arr_coc_model.persp_scorer = PerspectivalScorer()
    arr_coc_model.partic_scorer = ParticipatoryScorer()
    arr_coc_model.balancer = ContextualTensionBalancer()
    arr_coc_model.allocator = TokenAllocator()

    # Move to GPU
    device = arr_coc_model.model.device
    arr_coc_model.texture_gen = arr_coc_model.texture_gen.to(device)
    arr_coc_model.info_scorer = arr_coc_model.info_scorer.to(device)
    arr_coc_model.persp_scorer = arr_coc_model.persp_scorer.to(device)
    arr_coc_model.partic_scorer = arr_coc_model.partic_scorer.to(device)
    arr_coc_model.balancer = arr_coc_model.balancer.to(device)
    arr_coc_model.allocator = arr_coc_model.allocator.to(device)

    print("Models loaded successfully!")

def visualize_relevance(image, relevance_map, positions, budgets):
    """Create heatmap overlay visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Relevance heatmap
    relevance_np = relevance_map.cpu().numpy()
    axes[1].imshow(relevance_np, cmap='hot', interpolation='bilinear')
    axes[1].set_title("Relevance Map (ARR-COC)")
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(relevance_np, cmap='hot', alpha=0.5, interpolation='bilinear')

    # Plot selected positions
    positions_np = positions.cpu().numpy()
    budgets_np = budgets.cpu().numpy()

    # Normalize budgets for sizing
    sizes = (budgets_np - budgets_np.min()) / (budgets_np.max() - budgets_np.min() + 1e-8)
    sizes = 10 + 90 * sizes  # 10-100 point size

    axes[2].scatter(positions_np[:, 1], positions_np[:, 0],
                   s=sizes, c='cyan', alpha=0.6, edgecolors='white', linewidths=0.5)
    axes[2].set_title(f"Selected Positions (n={len(positions)})")
    axes[2].axis('off')

    plt.tight_layout()
    return fig

def compare_models(image, query):
    """Run both standard and ARR-COC models, return comparison"""
    global session_stats

    if arr_coc_model is None:
        return "Models not initialized!", None, None

    # Run standard Qwen3-VL
    print("Running standard Qwen3-VL...")
    result_standard = arr_coc_model.generate(image, query, use_arr_coc=False)

    # Run ARR-COC-enhanced
    print("Running ARR-COC-VIS...")
    result_arr_coc = arr_coc_model.generate(image, query, use_arr_coc=True)

    # Create visualization
    if result_arr_coc['relevance_map'] is not None:
        positions, budgets, _, _ = arr_coc_model.allocate_relevance(image, query)
        viz = visualize_relevance(
            image,
            result_arr_coc['relevance_map'],
            positions,
            budgets
        )
    else:
        viz = None

    # Format comparison
    comparison_md = f"""
## 📊 Comparison Results

### Standard Qwen3-VL
**Answer:** {result_standard['answer']}

**Stats:**
- ⏱️ Time: {result_standard['time_ms']:.2f}ms
- 💾 Memory: {result_standard['memory_mb']:.2f}MB
- 🔢 Patches: {result_standard['num_patches']}

---

### ARR-COC-VIS Enhanced
**Answer:** {result_arr_coc['answer']}

**Stats:**
- ⏱️ Time: {result_arr_coc['time_ms']:.2f}ms
- 💾 Memory: {result_arr_coc['memory_mb']:.2f}MB
- 🔢 Patches: {result_arr_coc['num_patches']}
- ⚖️ Tensions: {result_arr_coc['tensions']}
  - Compress ↔ Particularize: {result_arr_coc['tensions'][0]:.3f}
  - Exploit ↔ Explore: {result_arr_coc['tensions'][1]:.3f}
  - Focus ↔ Diversify: {result_arr_coc['tensions'][2]:.3f}

---

### 📈 Efficiency Gains
- ⚡ Speedup: {result_standard['time_ms'] / result_arr_coc['time_ms']:.2f}×
- 💾 Memory Reduction: {(1 - result_arr_coc['memory_mb'] / result_standard['memory_mb']) * 100:.1f}%
"""

    # Log to session stats
    session_stats.append({
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'standard_time_ms': result_standard['time_ms'],
        'arr_coc_time_ms': result_arr_coc['time_ms'],
        'speedup': result_standard['time_ms'] / result_arr_coc['time_ms'],
        'standard_memory_mb': result_standard['memory_mb'],
        'arr_coc_memory_mb': result_arr_coc['memory_mb'],
        'tensions': result_arr_coc['tensions']
    })

    return comparison_md, viz, create_stats_summary()

def create_stats_summary():
    """Create aggregate statistics visualization"""
    if len(session_stats) == 0:
        return None

    df = pd.DataFrame(session_stats)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Speedup over time
    axes[0, 0].plot(df['speedup'], marker='o')
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    axes[0, 0].set_title('Speedup Over Session')
    axes[0, 0].set_xlabel('Query Number')
    axes[0, 0].set_ylabel('Speedup (×)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Time comparison
    x = np.arange(len(df))
    width = 0.35
    axes[0, 1].bar(x - width/2, df['standard_time_ms'], width, label='Standard', alpha=0.8)
    axes[0, 1].bar(x + width/2, df['arr_coc_time_ms'], width, label='ARR-COC', alpha=0.8)
    axes[0, 1].set_title('Inference Time Comparison')
    axes[0, 1].set_xlabel('Query Number')
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Tension distribution
    if 'tensions' in df.columns and df['tensions'].iloc[0] is not None:
        tensions_array = np.array([t for t in df['tensions'] if t is not None])
        if len(tensions_array) > 0:
            axes[1, 0].boxplot(tensions_array, labels=['Compress\n↔\nParticularize',
                                                       'Exploit\n↔\nExplore',
                                                       'Focus\n↔\nDiversify'])
            axes[1, 0].set_title('Adaptive Tension Distribution')
            axes[1, 0].set_ylabel('Tension Value')
            axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics
    summary_text = f"""
Session Summary (n={len(df)})

Average Speedup: {df['speedup'].mean():.2f}×
Std Speedup: {df['speedup'].std():.2f}

Avg Time Reduction: {(df['standard_time_ms'] - df['arr_coc_time_ms']).mean():.2f}ms
Avg Memory Reduction: {(df['standard_memory_mb'] - df['arr_coc_memory_mb']).mean():.2f}MB

Best Speedup: {df['speedup'].max():.2f}×
Worst Speedup: {df['speedup'].min():.2f}×
"""
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()
    return fig

def clear_session():
    """Clear session statistics"""
    global session_stats
    session_stats = []
    return "Session cleared!", None

# Create Gradio interface
with gr.Blocks(title="ARR-COC-VIS Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 ARR-COC-VIS: Adaptive Relevance Realization for Vision-Language Models

    **Compare Standard Qwen3-VL vs ARR-COC-Enhanced Performance**

    Upload an image and ask a question to see:
    - Side-by-side answer comparison
    - Relevance allocation visualization
    - Efficiency metrics (time, memory)
    - Adaptive tension values (contextual strategy)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            query_input = gr.Textbox(label="Your Question", placeholder="What objects are in this image?")

            with gr.Row():
                submit_btn = gr.Button("🚀 Compare Models", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Session", variant="secondary")

            gr.Markdown("### 📝 Example Queries")
            gr.Examples(
                examples=[
                    ["test_images/sample1.jpg", "What objects are in this image?"],
                    ["test_images/sample2.jpg", "Describe the scene in detail."],
                    ["test_images/sample3.jpg", "What is the small text in the corner?"],
                ],
                inputs=[image_input, query_input]
            )

        with gr.Column(scale=2):
            comparison_output = gr.Markdown(label="Comparison Results")

    with gr.Row():
        relevance_viz = gr.Plot(label="Relevance Allocation Visualization")
        stats_viz = gr.Plot(label="Session Statistics")

    gr.Markdown("""
    ---
    ### 🔬 Understanding the Results

    **Relevance Map:** Shows where ARR-COC allocates attention (hot = high relevance)

    **Selected Positions:** Cyan dots sized by token budget (large = more detail)

    **Adaptive Tensions:** Context-dependent strategy values:
    - **Compress ↔ Particularize:** Low = preserve detail, High = compress/overview
    - **Exploit ↔ Explore:** Low = explore broadly, High = focus on known targets
    - **Focus ↔ Diversify:** Low = spread tokens, High = concentrate tokens

    **Key Hypothesis:** Learned relevance allocation improves efficiency without sacrificing accuracy.
    """)

    # Event handlers
    submit_btn.click(
        fn=compare_models,
        inputs=[image_input, query_input],
        outputs=[comparison_output, relevance_viz, stats_viz]
    )

    clear_btn.click(
        fn=clear_session,
        outputs=[comparison_output, stats_viz]
    )

# Initialize on startup
print("Initializing ARR-COC-VIS Demo...")
initialize_models()

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
```

---

## Phase 5: Training & Validation

### 5.1 Training Script

**File: `train.py`**

```python
"""
Training script for ARR-COC-VIS components.

Three-stage curriculum:
1. Stage 1: Propositional + Perspectival (proxy loss with bounding boxes)
2. Stage 2: Participatory (VQA accuracy loss)
3. Stage 3: Procedural (efficiency + accuracy)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO: Implement full training loop
# Key components:
# - COCO dataset with bounding boxes (Stage 1)
# - VQAv2 dataset (Stage 2)
# - Adversarial examples (Stage 3)
# - Proxy losses vs end-to-end backprop
# - LoRA for Qwen3-VL (optional fine-tuning)
```

### 5.2 Evaluation Benchmarks

**File: `evaluation/benchmark.py`**

```python
"""
Comprehensive evaluation on multiple benchmarks:
- VQAv2 (accuracy)
- GQA (compositional reasoning)
- TextVQA (OCR capability)
- COCO Captioning (generation quality)
"""

# TODO: Implement benchmark evaluation
```

---

## Key Hypotheses to Validate

### Hypothesis 1: Multi-way knowing improves relevance detection
**Test:** Ablate info/persp/partic scorers, measure IoU with human attention

### Hypothesis 2: Adaptive tensions beat fixed tensions
**Test:** Compare fixed vs contextual balancer on diverse queries (Part 37)

### Hypothesis 3: Variable allocation improves efficiency
**Test:** Measure speedup and memory reduction vs uniform grid

### Hypothesis 4: Learned allocation matches task structure
**Test:** Analyze tension patterns across query types (specific vs vague)

### Hypothesis 5: System generalizes to new domains
**Test:** Evaluate on held-out categories (medical, satellite, etc.)

---

## Success Criteria

### Must Have (MVP):
✅ App.py runs locally
✅ Side-by-side comparison works
✅ Relevance visualization displays
✅ Basic efficiency metrics (time, memory)

### Should Have (Full System):
✅ Adaptive tensions functional
✅ Training pipeline implemented
✅ 2× speedup over baseline
✅ Comparable accuracy to standard Qwen

### Nice to Have (Future Work):
⭕ 40-channel full texture array
⭕ True sparse sampling (Strategy 1)
⭕ Multi-fixation protocol
⭕ Video understanding with temporal cache

---

## Implementation Sequence

1. **Clone repo, set up environment**
2. **Test baseline Qwen3-VL**
3. **Implement texture_array.py MVP**
4. **Implement knowing.py scorers**
5. **Implement balancing.py + attending.py**
6. **Integrate with Qwen (qwen_integration.py)**
7. **Build app.py interface**
8. **Test on example images**
9. **Iterate based on results**

---

**END OF PART 38 ADDENDUM**

*From philosophical dialogue to runnable code. The bridge is built. Now we cross it.*

∿◇∿
