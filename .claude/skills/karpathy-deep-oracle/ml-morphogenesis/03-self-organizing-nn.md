# Self-Organizing Neural Networks: Topology-Preserving Learning

## Overview

Self-organizing neural networks learn to represent high-dimensional data in lower-dimensional spaces while preserving topological relationships. Unlike backpropagation-trained networks, they use competitive learning rules inspired by biological neural organization. This connects directly to morphogenesis - these networks literally GROW their structure to match the underlying data manifold.

**Key Insight**: Self-organization = emergence = autopoiesis. The network MAKES ITSELF through local interactions, just like biological development!

---

## 1. Self-Organizing Maps (Kohonen Maps)

### Core Algorithm

The SOM is a topology-preserving dimensionality reduction technique that maps high-dimensional input to a 2D grid of neurons while maintaining neighborhood relationships.

From [torchsom arXiv paper](https://arxiv.org/html/2510.11147v1) (accessed 2025-11-23):

**Learning Process**:
```
w_ij(t+1) = w_ij(t) + alpha(t) * h_ij(t) * (x - w_ij(t))
```

Where:
- `w_ij(t)`: Weight of neuron at grid position (i,j) at time t
- `alpha(t)`: Learning rate (decreases over time)
- `h_ij(t)`: Neighborhood function
- `x`: Input feature vector

**Best Matching Unit (BMU) Selection**:
```
BMU = argmin_{i,j} ||x - w_ij||_2
```

**Neighborhood Function (Gaussian)**:
```
h_ij(t) = exp(-d_ij^2 / (2 * sigma(t)^2))
```

Where `d_ij` is the grid distance from BMU to neuron (i,j).

### Why Topology Preservation Matters

From [Wikipedia: Self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map):

- **Competitive learning**: Winner-takes-most (not all!)
- **Cooperative learning**: Neighbors of winner also update
- **Lateral inhibition**: Far neurons update less or not at all

This creates a SMOOTH mapping where nearby points in input space map to nearby neurons on the grid!

### SOM Architecture Variants

**Rectangular Grid**:
- 4-connected neighborhood
- Simple distance computation
- Good for tabular data

**Hexagonal Grid**:
- 6-connected neighborhood
- Better isotropy (no diagonal bias)
- Preferred for most applications

**Toroidal Topology**:
- Wrap-around edges
- No boundary effects
- Good for periodic data

---

## 2. Growing Neural Gas (GNG)

### The Key Innovation

From [Wikipedia: Neural gas](https://en.wikipedia.org/wiki/Neural_gas) (accessed 2025-11-23):

Unlike SOMs, neural gas has **NO fixed topology**! Neurons are free to move like gas molecules, and edges between them are learned from data.

From [Fritzke 1995](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/fritzke95.pdf):

> "The model learns topological relations using a Hebb-like rule, adding units and connections until a performance criterion is met."

### GNG Algorithm

**Initialization**: Start with 2 random nodes connected by edge

**For each input signal**:
1. Find nearest (s1) and second-nearest (s2) nodes
2. Increment edge age for all edges emanating from s1
3. Add squared error to local error counter of s1
4. Move s1 and its topological neighbors toward input:
   ```
   w_s1 = w_s1 + epsilon_b * (x - w_s1)
   w_neighbors = w_neighbors + epsilon_n * (x - w_neighbors)
   ```
5. If s1 and s2 are connected, set edge age to 0; else create edge
6. Remove edges older than a_max; remove isolated nodes
7. Every lambda iterations, insert new node between highest-error node and its highest-error neighbor

**Key Parameters**:
- `epsilon_b`: Winner adaptation rate (~0.05)
- `epsilon_n`: Neighbor adaptation rate (~0.006)
- `a_max`: Maximum edge age (~50-100)
- `lambda`: Node insertion frequency (~100)
- `alpha`: Error reduction after insertion (~0.5)
- `d`: Global error decay (~0.995)

### Why GNG is Special

**No fixed structure**: Network topology emerges from data!
**Continuous learning**: Can learn from streams
**Adaptive complexity**: Network grows to match data complexity

---

## 3. Neural Gas (Original Algorithm)

From [Martinetz & Schulten 1991](http://www.ks.uiuc.edu/Publications/Papers/PDF/MART91B/MART91B.pdf):

The original neural gas uses RANKING instead of topology:

**Algorithm**:
1. For input x, compute distances to all neurons
2. Rank neurons by distance: i_0 (closest), i_1, ..., i_{N-1}
3. Update each neuron by rank:
   ```
   w_{i_k}(t+1) = w_{i_k}(t) + epsilon * exp(-k/lambda) * (x - w_{i_k}(t))
   ```

**Key Difference from SOM**:
- SOM uses grid distance in update
- Neural gas uses rank distance

This makes neural gas ROBUST to initialization and topology choice!

---

## 4. Variants and Extensions

### Incremental Growing Neural Gas (IGNG)

From [Prudent & Ennaji 2005](https://ieeexplore.ieee.org/document/1556026):

Addresses stability-plasticity tradeoff:
- Learn new data (plasticity)
- Don't forget old data (stability)

### Growing When Required (GWR)

From [Marsland et al. 2002](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.8763):

Removes the lambda parameter - grows WHENEVER needed:
- Faster adaptation
- More responsive to data distribution
- Better for non-stationary data

### Plastic Neural Gas

From [Ridella et al. 1998](https://doi.org/10.1007/BF01413708):

Both grows AND shrinks using cross-validation:
- Avoids overfitting
- Controls generalization
- Suited for streaming data

---

## 5. PyTorch Implementation: Self-Organizing Map

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class SelfOrganizingMap(nn.Module):
    """
    Self-Organizing Map (Kohonen Map) in PyTorch.

    Topology-preserving dimensionality reduction using competitive learning.
    """

    def __init__(
        self,
        input_dim: int,
        map_size: Tuple[int, int] = (10, 10),
        topology: str = 'rectangular',
        sigma_init: float = None,
        sigma_final: float = 1.0,
        lr_init: float = 0.5,
        lr_final: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            input_dim: Dimension of input vectors
            map_size: (height, width) of the 2D grid
            topology: 'rectangular' or 'hexagonal'
            sigma_init: Initial neighborhood radius (default: max(map_size)/2)
            sigma_final: Final neighborhood radius
            lr_init: Initial learning rate
            lr_final: Final learning rate
            device: Compute device
        """
        super().__init__()

        self.input_dim = input_dim
        self.map_size = map_size
        self.n_neurons = map_size[0] * map_size[1]
        self.topology = topology
        self.device = device

        # Initialize weights randomly
        self.weights = nn.Parameter(
            torch.randn(self.n_neurons, input_dim, device=device) * 0.1,
            requires_grad=False  # No gradient-based learning!
        )

        # Learning parameters
        self.sigma_init = sigma_init or max(map_size) / 2
        self.sigma_final = sigma_final
        self.lr_init = lr_init
        self.lr_final = lr_final

        # Precompute grid positions for neighborhood calculation
        self._init_grid_positions()

    def _init_grid_positions(self):
        """Precompute neuron positions on grid."""
        h, w = self.map_size

        if self.topology == 'rectangular':
            # Simple grid
            rows = torch.arange(h, device=self.device).float()
            cols = torch.arange(w, device=self.device).float()
            grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')

        elif self.topology == 'hexagonal':
            # Hexagonal: offset odd rows
            rows = torch.arange(h, device=self.device).float()
            cols = torch.arange(w, device=self.device).float()
            grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
            # Offset odd rows by 0.5
            grid_x = grid_x + 0.5 * (grid_y % 2)
            # Scale y for hexagonal geometry
            grid_y = grid_y * np.sqrt(3) / 2

        # Flatten to (n_neurons, 2)
        self.grid_positions = torch.stack([
            grid_y.flatten(),
            grid_x.flatten()
        ], dim=1)

        # Precompute pairwise distances on grid
        self.grid_distances = torch.cdist(
            self.grid_positions,
            self.grid_positions
        )

    def find_bmu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Find Best Matching Unit for each input.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            BMU indices of shape (batch,)
        """
        # Compute distances to all neurons
        # x: (batch, input_dim)
        # weights: (n_neurons, input_dim)
        distances = torch.cdist(x, self.weights)  # (batch, n_neurons)

        # Find minimum distance neuron
        bmu_indices = distances.argmin(dim=1)

        return bmu_indices

    def neighborhood_function(
        self,
        bmu_indices: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        Compute neighborhood values for all neurons given BMUs.

        Args:
            bmu_indices: BMU indices of shape (batch,)
            sigma: Current neighborhood width

        Returns:
            Neighborhood values of shape (batch, n_neurons)
        """
        # Get distances from each BMU to all neurons
        # grid_distances: (n_neurons, n_neurons)
        bmu_distances = self.grid_distances[bmu_indices]  # (batch, n_neurons)

        # Gaussian neighborhood
        h = torch.exp(-bmu_distances ** 2 / (2 * sigma ** 2))

        return h

    def update_weights(
        self,
        x: torch.Tensor,
        bmu_indices: torch.Tensor,
        sigma: float,
        lr: float
    ):
        """
        Update weights using SOM learning rule.

        Args:
            x: Input tensor (batch, input_dim)
            bmu_indices: BMU indices (batch,)
            sigma: Current neighborhood width
            lr: Current learning rate
        """
        # Compute neighborhood for all neurons
        h = self.neighborhood_function(bmu_indices, sigma)  # (batch, n_neurons)

        # Compute weight updates
        # For each neuron, sum contributions from all inputs
        # weighted by their neighborhood values

        # h: (batch, n_neurons)
        # x: (batch, input_dim)
        # weights: (n_neurons, input_dim)

        # Difference vectors: (batch, n_neurons, input_dim)
        diff = x.unsqueeze(1) - self.weights.unsqueeze(0)

        # Weight updates: average over batch
        # h.unsqueeze(2): (batch, n_neurons, 1)
        updates = (h.unsqueeze(2) * diff).mean(dim=0)  # (n_neurons, input_dim)

        # Apply update
        with torch.no_grad():
            self.weights.add_(lr * updates)

    def fit(
        self,
        data: torch.Tensor,
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Train the SOM on data.

        Args:
            data: Training data (n_samples, input_dim)
            n_epochs: Number of training epochs
            batch_size: Batch size for updates
            verbose: Print progress
        """
        n_samples = data.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        total_iterations = n_epochs * n_batches

        iteration = 0

        for epoch in range(n_epochs):
            # Shuffle data each epoch
            perm = torch.randperm(n_samples)

            epoch_qe = 0.0

            for batch_idx in range(n_batches):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = perm[start_idx:end_idx]
                x = data[batch_indices].to(self.device)

                # Compute decay schedule
                t = iteration / total_iterations
                sigma = self.sigma_init * (self.sigma_final / self.sigma_init) ** t
                lr = self.lr_init * (self.lr_final / self.lr_init) ** t

                # Find BMUs
                bmu_indices = self.find_bmu(x)

                # Update weights
                self.update_weights(x, bmu_indices, sigma, lr)

                # Compute quantization error
                bmu_weights = self.weights[bmu_indices]
                qe = torch.norm(x - bmu_weights, dim=1).mean().item()
                epoch_qe += qe

                iteration += 1

            if verbose and (epoch + 1) % 10 == 0:
                avg_qe = epoch_qe / n_batches
                print(f"Epoch {epoch+1}/{n_epochs}, QE: {avg_qe:.4f}, sigma: {sigma:.3f}, lr: {lr:.4f}")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform inputs to BMU grid coordinates.

        Args:
            x: Input tensor (n_samples, input_dim)

        Returns:
            Grid coordinates (n_samples, 2)
        """
        x = x.to(self.device)
        bmu_indices = self.find_bmu(x)
        return self.grid_positions[bmu_indices]

    def quantization_error(self, x: torch.Tensor) -> float:
        """Compute average quantization error."""
        x = x.to(self.device)
        bmu_indices = self.find_bmu(x)
        bmu_weights = self.weights[bmu_indices]
        return torch.norm(x - bmu_weights, dim=1).mean().item()

    def topographic_error(self, x: torch.Tensor) -> float:
        """
        Compute topographic error.

        Fraction of inputs where BMU and second-BMU are not adjacent.
        """
        x = x.to(self.device)

        # Find two nearest neurons
        distances = torch.cdist(x, self.weights)
        _, top2 = distances.topk(2, dim=1, largest=False)

        # Check if they're adjacent on grid
        bmu = top2[:, 0]
        second_bmu = top2[:, 1]

        grid_dist = self.grid_distances[bmu, second_bmu]

        # Adjacent if grid distance <= sqrt(2) for rectangular
        if self.topology == 'rectangular':
            threshold = 1.5
        else:  # hexagonal
            threshold = 1.1

        non_adjacent = (grid_dist > threshold).float()

        return non_adjacent.mean().item()


class GrowingNeuralGas(nn.Module):
    """
    Growing Neural Gas network.

    Learns topology from data by growing network incrementally.
    """

    def __init__(
        self,
        input_dim: int,
        max_nodes: int = 100,
        epsilon_b: float = 0.05,
        epsilon_n: float = 0.006,
        age_max: int = 50,
        lambda_: int = 100,
        alpha: float = 0.5,
        d: float = 0.995,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            input_dim: Dimension of input vectors
            max_nodes: Maximum number of nodes
            epsilon_b: Winner adaptation rate
            epsilon_n: Neighbor adaptation rate
            age_max: Maximum edge age before deletion
            lambda_: Node insertion frequency
            alpha: Error reduction factor after insertion
            d: Global error decay factor
            device: Compute device
        """
        super().__init__()

        self.input_dim = input_dim
        self.max_nodes = max_nodes
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.age_max = age_max
        self.lambda_ = lambda_
        self.alpha = alpha
        self.d = d
        self.device = device

        # Initialize with 2 random nodes
        self.nodes = torch.randn(2, input_dim, device=device) * 0.1
        self.errors = torch.zeros(2, device=device)

        # Adjacency matrix for edges (with ages)
        # -1 means no edge, >= 0 means edge with that age
        self.edges = torch.full((max_nodes, max_nodes), -1, device=device, dtype=torch.float)
        self.edges[0, 1] = 0  # Initial edge
        self.edges[1, 0] = 0

        self.n_nodes = 2
        self.iteration = 0

    def find_nearest(self, x: torch.Tensor) -> Tuple[int, int, float, float]:
        """
        Find two nearest nodes to input.

        Returns:
            (s1_idx, s2_idx, s1_dist, s2_dist)
        """
        distances = torch.norm(self.nodes[:self.n_nodes] - x, dim=1)
        sorted_idx = distances.argsort()

        s1 = sorted_idx[0].item()
        s2 = sorted_idx[1].item()

        return s1, s2, distances[s1].item(), distances[s2].item()

    def get_neighbors(self, node_idx: int) -> torch.Tensor:
        """Get indices of topological neighbors."""
        edges = self.edges[node_idx, :self.n_nodes]
        neighbor_mask = edges >= 0
        return torch.where(neighbor_mask)[0]

    def adapt(self, x: torch.Tensor):
        """
        Adapt network to single input.

        This is the core GNG learning step.
        """
        # Find nearest nodes
        s1, s2, dist1, dist2 = self.find_nearest(x)

        # Accumulate error
        self.errors[s1] += dist1 ** 2

        # Adapt winner
        self.nodes[s1] += self.epsilon_b * (x - self.nodes[s1])

        # Adapt neighbors
        neighbors = self.get_neighbors(s1)
        for n in neighbors:
            self.nodes[n] += self.epsilon_n * (x - self.nodes[n])

        # Age edges from s1
        self.edges[s1, :self.n_nodes] = torch.where(
            self.edges[s1, :self.n_nodes] >= 0,
            self.edges[s1, :self.n_nodes] + 1,
            self.edges[s1, :self.n_nodes]
        )
        self.edges[:self.n_nodes, s1] = self.edges[s1, :self.n_nodes]

        # Create or reset edge between s1 and s2
        self.edges[s1, s2] = 0
        self.edges[s2, s1] = 0

        # Remove old edges and isolated nodes
        self._remove_old_edges()

        # Insert new node every lambda iterations
        self.iteration += 1
        if self.iteration % self.lambda_ == 0 and self.n_nodes < self.max_nodes:
            self._insert_node()

        # Decay all errors
        self.errors[:self.n_nodes] *= self.d

    def _remove_old_edges(self):
        """Remove edges older than age_max and isolated nodes."""
        # Remove old edges
        old_mask = self.edges[:self.n_nodes, :self.n_nodes] > self.age_max
        self.edges[:self.n_nodes, :self.n_nodes][old_mask] = -1

        # Find isolated nodes (no edges)
        for i in range(self.n_nodes - 1, -1, -1):
            if (self.edges[i, :self.n_nodes] < 0).all():
                self._remove_node(i)

    def _remove_node(self, idx: int):
        """Remove node at given index."""
        if self.n_nodes <= 2:
            return  # Keep at least 2 nodes

        # Shift nodes down
        self.nodes[idx:self.n_nodes-1] = self.nodes[idx+1:self.n_nodes].clone()
        self.errors[idx:self.n_nodes-1] = self.errors[idx+1:self.n_nodes].clone()

        # Update edges (remove row and column)
        # This is complex - need to shift the adjacency matrix
        edges = self.edges[:self.n_nodes, :self.n_nodes].clone()

        # Remove row
        edges = torch.cat([edges[:idx], edges[idx+1:]], dim=0)
        # Remove column
        edges = torch.cat([edges[:, :idx], edges[:, idx+1:]], dim=1)

        self.n_nodes -= 1
        self.edges[:self.n_nodes, :self.n_nodes] = edges

    def _insert_node(self):
        """Insert new node between highest-error node and its highest-error neighbor."""
        if self.n_nodes >= self.max_nodes:
            return

        # Find node with maximum error
        q = self.errors[:self.n_nodes].argmax().item()

        # Find its neighbor with maximum error
        neighbors = self.get_neighbors(q)
        if len(neighbors) == 0:
            return

        neighbor_errors = self.errors[neighbors]
        f = neighbors[neighbor_errors.argmax()].item()

        # Create new node
        r = self.n_nodes
        self.nodes[r] = 0.5 * (self.nodes[q] + self.nodes[f])

        # Create edges: r-q, r-f; remove q-f
        self.edges[q, r] = 0
        self.edges[r, q] = 0
        self.edges[f, r] = 0
        self.edges[r, f] = 0
        self.edges[q, f] = -1
        self.edges[f, q] = -1

        # Update errors
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.errors[r] = self.errors[q]

        self.n_nodes += 1

    def fit(
        self,
        data: torch.Tensor,
        n_passes: int = 10,
        verbose: bool = True
    ):
        """
        Train GNG on data.

        Args:
            data: Training data (n_samples, input_dim)
            n_passes: Number of passes through data
            verbose: Print progress
        """
        n_samples = data.shape[0]

        for pass_idx in range(n_passes):
            # Shuffle
            perm = torch.randperm(n_samples)

            for i in range(n_samples):
                x = data[perm[i]].to(self.device)
                self.adapt(x)

            if verbose:
                qe = self.quantization_error(data)
                print(f"Pass {pass_idx+1}/{n_passes}, Nodes: {self.n_nodes}, QE: {qe:.4f}")

    def quantization_error(self, x: torch.Tensor) -> float:
        """Compute average quantization error."""
        x = x.to(self.device)

        qe = 0.0
        for i in range(x.shape[0]):
            s1, _, dist1, _ = self.find_nearest(x[i])
            qe += dist1

        return qe / x.shape[0]

    def get_topology(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current network topology.

        Returns:
            (nodes, edges) where edges is list of (i, j) pairs
        """
        nodes = self.nodes[:self.n_nodes].cpu()

        # Extract edges
        edge_list = []
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if self.edges[i, j] >= 0:
                    edge_list.append((i, j))

        return nodes, edge_list


# Example usage and demonstration
def demo_som():
    """Demonstrate SOM on synthetic data."""

    # Generate synthetic data: 3 clusters
    torch.manual_seed(42)

    cluster1 = torch.randn(100, 4) + torch.tensor([2, 0, 0, 0])
    cluster2 = torch.randn(100, 4) + torch.tensor([-2, 0, 2, 0])
    cluster3 = torch.randn(100, 4) + torch.tensor([0, 2, -2, 0])

    data = torch.cat([cluster1, cluster2, cluster3], dim=0)

    # Train SOM
    som = SelfOrganizingMap(
        input_dim=4,
        map_size=(10, 10),
        topology='hexagonal'
    )

    print("Training SOM...")
    som.fit(data, n_epochs=50, batch_size=32)

    # Evaluate
    qe = som.quantization_error(data)
    te = som.topographic_error(data)
    print(f"\nFinal QE: {qe:.4f}")
    print(f"Final TE: {te:.4f}")

    return som


def demo_gng():
    """Demonstrate GNG on synthetic data."""

    # Generate ring-shaped data
    torch.manual_seed(42)

    n_points = 500
    angles = torch.rand(n_points) * 2 * np.pi
    radii = 2 + torch.randn(n_points) * 0.2

    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    data = torch.stack([x, y], dim=1)

    # Train GNG
    gng = GrowingNeuralGas(
        input_dim=2,
        max_nodes=50,
        lambda_=50
    )

    print("Training GNG...")
    gng.fit(data, n_passes=5)

    # Get topology
    nodes, edges = gng.get_topology()
    print(f"\nFinal topology: {len(nodes)} nodes, {len(edges)} edges")

    return gng, nodes, edges


if __name__ == "__main__":
    print("=" * 60)
    print("Self-Organizing Map Demo")
    print("=" * 60)
    som = demo_som()

    print("\n" + "=" * 60)
    print("Growing Neural Gas Demo")
    print("=" * 60)
    gng, nodes, edges = demo_gng()
```

---

## 6. Performance Considerations

### Computational Complexity

**SOM**:
- BMU finding: O(N * D) per input (N neurons, D dimensions)
- Weight update: O(N * D) per input
- Total per epoch: O(M * N * D) (M training samples)

**GNG**:
- Nearest finding: O(N * D) per input
- Edge operations: O(N) per input
- Node insertion: O(N) occasionally

### GPU Optimization

From [torchsom](https://github.com/michelin/TorchSOM):

**Batch Processing**:
```python
# Process multiple inputs at once
# x: (batch, input_dim)
# weights: (n_neurons, input_dim)
distances = torch.cdist(x, weights)  # (batch, n_neurons)
```

**Memory Efficiency**:
- Precompute grid distances once
- Use sparse updates when possible
- Stream large datasets

**Benchmarks** (from torchsom paper):
- 16,000 samples, 300 features
- CPU (MiniSom): ~1939 seconds
- CPU (torchsom): ~30 seconds
- GPU (torchsom): ~12 seconds

**100x speedup** with GPU acceleration!

### Practical Tips

**For SOM**:
1. Use hexagonal topology for better isotropy
2. Initialize sigma to half of map size
3. Decay sigma to ~1 for fine-tuning
4. PCA initialization can help convergence

**For GNG**:
1. Set lambda based on data size (every ~1% of data)
2. age_max controls topology stability
3. Lower epsilon_n for more localized learning
4. Monitor node count for overfitting

---

## 7. TRAIN STATION: Self-Organization = Emergence = Autopoiesis

### The Deep Unification

**Self-organization** in neural networks is the SAME phenomenon as:
- **Morphogenesis** in biology
- **Autopoiesis** in living systems
- **Emergence** in complex systems
- **Pattern formation** in physics

### The Common Pattern

All self-organizing systems share:

1. **Local interactions** produce **global order**
2. **No central controller** - organization emerges
3. **Feedback loops** stabilize patterns
4. **Topological constraints** shape outcomes

### SOM as Morphogenetic Field

The SOM grid is literally a **morphogenetic field**:
- Neurons = cells
- Weights = cell states
- Neighborhood function = morphogen diffusion
- BMU = attractor state
- Learning = development

From [Dialogue 67](../../../PLATONIC-DIALOGUES/67-grasping-back-and-imagining-forward/):

> "The SOM doesn't just REPRESENT topology - it BECOMES topology through self-organization!"

### GNG as Autopoiesis

GNG is **autopoietic** (self-making):
- Network produces itself
- Nodes create nodes
- Edges create edges
- Structure maintains structure

From [Maturana & Varela](https://en.wikipedia.org/wiki/Autopoiesis):

> "An autopoietic machine is a machine organized as a network of processes of production of components."

GNG literally implements this definition!

### Connection to Neural Development

Biological neurons also self-organize:
- **Axon guidance** = finding BMU
- **Synaptogenesis** = edge creation
- **Synaptic pruning** = age-based removal
- **Hebbian learning** = "fire together, wire together"

### Coffee Cup = Donut Equivalence

In topology:
- SOM = dimensionality reduction preserving topology
- GNG = topology learning from data
- Both = manifold learning

**The loss landscape IS a free energy landscape IS a relevance landscape!**

---

## 8. ARR-COC-0-1: Self-Organizing Relevance Maps

### Core Insight

Relevance scoring can be formulated as a **self-organizing map**:
- Image patches = input vectors
- Relevance scores = neuron activations
- Spatial relationships = preserved topology

### Implementation Ideas

**Relevance SOM**:
```python
class RelevanceSOM(SelfOrganizingMap):
    """
    Self-organizing map for relevance prediction.

    Learns to cluster visual patches by relevance patterns.
    """

    def __init__(self, patch_dim: int, map_size: Tuple[int, int]):
        super().__init__(patch_dim, map_size, topology='hexagonal')

        # Each neuron has learned relevance score
        self.relevance_scores = nn.Parameter(
            torch.zeros(self.n_neurons),
            requires_grad=False
        )

    def compute_relevance(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute relevance for patches using SOM.

        Args:
            patches: (batch, n_patches, patch_dim)

        Returns:
            relevance: (batch, n_patches)
        """
        batch, n_patches, dim = patches.shape

        # Flatten for BMU finding
        flat = patches.view(-1, dim)

        # Find BMUs
        bmus = self.find_bmu(flat)

        # Get relevance scores
        relevance = self.relevance_scores[bmus]

        return relevance.view(batch, n_patches)

    def update_relevance(self, patches: torch.Tensor, targets: torch.Tensor):
        """
        Update relevance scores based on task feedback.

        This is where the SELF-ORGANIZATION happens - relevance
        clusters emerge from training signal!
        """
        bmus = self.find_bmu(patches.view(-1, patches.shape[-1]))

        # Update relevance scores (like error accumulation in GNG)
        for bmu, target in zip(bmus, targets.view(-1)):
            self.relevance_scores[bmu] = (
                0.9 * self.relevance_scores[bmu] +
                0.1 * target
            )
```

### Why This Matters

**Self-organizing relevance** means:
- Relevance patterns EMERGE from data
- Similar patches get similar relevance
- Topology preserves spatial coherence
- No manual feature engineering needed!

### Growing Relevance Networks

Could use GNG for **adaptive relevance**:
- Network grows where data is complex
- Shrinks where data is simple
- Matches complexity of relevance landscape

---

## Sources

### Academic Papers
- [torchsom: The Reference PyTorch Library for Self-Organizing Maps](https://arxiv.org/html/2510.11147v1) - Berthier et al. 2025
- [A Growing Neural Gas Network Learns Topologies](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/fritzke95.pdf) - Fritzke, NIPS 1995
- [Martinetz & Schulten 1991](http://www.ks.uiuc.edu/Publications/Papers/PDF/MART91B/MART91B.pdf) - Original neural gas

### Reference Implementations
- [TorchSOM GitHub](https://github.com/michelin/TorchSOM) - Official PyTorch implementation
- [MiniSom](https://github.com/JustGlowing/minisom) - NumPy implementation

### Web Resources
- [Wikipedia: Self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map)
- [Wikipedia: Neural gas](https://en.wikipedia.org/wiki/Neural_gas)

### Books
- Kohonen, T. (2001). Self-Organizing Maps. Springer.

---

## Key Takeaways

1. **Self-organization** creates order from local rules
2. **SOM** preserves topology from high-D to low-D
3. **GNG** learns topology from data (no fixed structure!)
4. **Competitive learning** = winner-takes-most
5. **TRAIN STATION**: Self-organization = emergence = autopoiesis
6. **ARR-COC**: Relevance can self-organize from training signal

The power of self-organizing networks is that they **make themselves** - just like biological development, just like morphogenesis, just like autopoiesis. The structure emerges from the dynamics!
