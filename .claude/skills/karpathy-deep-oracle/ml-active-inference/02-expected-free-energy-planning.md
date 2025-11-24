# Expected Free Energy Planning

## Overview

Expected Free Energy (EFE) is the central quantity that active inference agents minimize through action selection. Unlike variational free energy which bounds current model evidence, EFE extends planning into uncertain futures by balancing **epistemic value** (information gain) with **pragmatic value** (goal achievement). This creates agents that naturally resolve the exploration-exploitation dilemma.

**The Core Insight**: EFE decomposes into two terms:
- **Extrinsic Value**: Expected log-likelihood of desired observations (reward-seeking)
- **Epistemic Value**: Expected information gain (uncertainty-reducing exploration)

This dual nature makes EFE agents intrinsically motivated to explore unknown regions while pursuing goals - no ad hoc exploration bonuses needed.

---

## Section 1: EFE Computation

### The Mathematical Foundation

The Expected Free Energy for a policy pi at future time tau is:

```
G(pi) = E_Q(o,x|pi)[ln Q(x|pi) - ln p~(o,x)]
```

Where:
- `Q(x|pi)` = variational prior (predicted states given policy)
- `p~(o,x)` = biased generative model (desired observations)
- `o` = observations, `x` = hidden states

### Key Decomposition: Extrinsic + Epistemic

```
G(pi) = -E_Q[ln p~(o)]          # Extrinsic: maximize desired observation probability
        - E_Q[D_KL[Q(x|o)||Q(x|pi)]]  # Epistemic: maximize information gain
```

**Extrinsic Value**: Drives agent toward preferred observations (goals/rewards)
**Epistemic Value**: Drives agent to reduce uncertainty by visiting informative states

### Contrast with Variational Free Energy (VFE)

The VFE at current time t bounds model evidence:
```
F = E_Q(x|o)[ln Q(x|o) - ln p(o,x)]
```

Key difference: VFE uses **posterior** Q(x|o), EFE uses **prior** Q(x|pi). This swap creates the information gain term!

### The Free Energy of the Future (FEF)

A more direct extension of VFE to the future:
```
FEF = E_Q(o,x|pi)[ln Q(x|o) - ln p~(o,x)]
```

Critically, FEF has **positive** information gain (discourages exploration), while EFE has **negative** information gain (encourages exploration). The relationship:

```
EFE = FEF - Information_Gain
```

This shows exploration doesn't arise "naturally" from extending VFE - it's added by construction in EFE!

From [Millidge et al. 2021 "Whence the Expected Free Energy?"](https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy) (Neural Computation):
- EFE is not simply "free energy in the future"
- The epistemic term arises from using prior instead of posterior
- Proposes Free Energy of Expected Future (FEEF) as alternative

---

## Section 2: Epistemic + Pragmatic Value

### The Exploration-Exploitation Decomposition

The EFE can be decomposed multiple ways, but the most important separates:

**Pragmatic (Instrumental) Value**:
```python
pragmatic = -E_Q[ln p~(o)]  # Expected log-prob of desired observations
```

**Epistemic (Intrinsic) Value**:
```python
epistemic = -E_Q[D_KL[Q(x|o) || Q(x|pi)]]  # Expected information gain
```

### Why This Matters

1. **No Ad Hoc Exploration Bonuses**: Unlike RL methods that add curiosity/entropy bonuses, EFE derives exploration from first principles

2. **Automatic Scheduling**: Epistemic value dominates early (high uncertainty), pragmatic value dominates late (low uncertainty)

3. **Context-Sensitive Exploration**: Only explores in task-relevant dimensions

### Alternative Decomposition: Risk and Ambiguity

Using factorization `p~(o,x) = p(o|x)p~(x)`:

```
G(pi) = E_Q[H[p(o|x)]]        # Ambiguity: entropy of likelihood
        + D_KL[Q(x|pi)||p~(x)] # Risk: divergence from desired states
```

**Ambiguity**: Avoid states with uncertain observation mapping
**Risk**: Stay close to preferred state distribution

### Epistemic Value as Bayesian Surprise

The epistemic term equals expected Bayesian surprise - the KL divergence between posterior and prior beliefs. This connects to:
- **Curiosity**: Systems seek surprising observations
- **Salience**: Attention drawn to prediction-violating stimuli
- **Learning**: Maximum information gain drives efficient learning

From [Friston et al. 2015 "Active Inference and Epistemic Value"](https://www.tandfonline.com/doi/full/10.1080/17588928.2015.1020053) (Cognitive Neuroscience):
- Cited by 920+ papers
- Introduces variational formulation of explorative behavior
- Shows epistemic value maximized until no further information gain

---

## Section 3: Tree Search with EFE

### Monte Carlo Tree Search for Active Inference

The key insight from [Fountas et al. 2020](https://proceedings.neurips.cc/paper/2020/file/865dfbde8a344b44095495f3591f7407-Paper.pdf) (NeurIPS):
MCTS can find free-energy-optimal policies by treating EFE as the value to maximize.

**MCTS-Active Inference Algorithm**:

1. **Selection**: Navigate tree using UCB-like criterion based on EFE
2. **Expansion**: Add new nodes for unexplored actions
3. **Simulation**: Roll out using habitual network
4. **Backpropagation**: Update EFE estimates up the tree

### The UCB-EFE Connection

Standard UCB: `Q(a) + c * sqrt(ln(N)/n(a))`

EFE naturally provides both terms:
- Q(a) ~ pragmatic value (exploit)
- Uncertainty term ~ epistemic value (explore)

**This is why EFE = UCB = Thompson Sampling!** (see Train Station section)

### Active Inference Tree Search (AITS)

From [Maisto et al. 2025](https://www.sciencedirect.com/science/article/pii/S0925231224020903) (Neurocomputing):

```python
def active_inference_tree_search(root, horizon, simulations):
    for _ in range(simulations):
        node = root
        path = []

        # Selection: descend tree using EFE
        while node.is_expanded():
            action = select_action_efe(node)
            path.append((node, action))
            node = node.children[action]

        # Expansion: add child nodes
        if not node.is_terminal():
            expand_node(node, model)

        # Simulation: compute EFE for new node
        efe = compute_expected_free_energy(node, horizon)

        # Backpropagation: update tree
        for parent, action in reversed(path):
            parent.update_efe(action, efe)

    # Return best policy
    return root.best_action()
```

### Boosting MCTS with Free Energy Minimization

From [Dao et al. 2025](https://direct.mit.edu/neco/article/37/12/2205/133239/Boosting-MCTS-With-Free-Energy-Minimization) (Neural Computation):

Novel framework integrating MCTS with active inference:
- Uses EFE to guide tree expansion
- Epistemic value reduces branching factor
- Achieves better exploration than pure UCB

---

## Section 4: PyTorch Implementation - Planning with Expected Free Energy

### Complete EFE Planning Agent

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import List, Tuple, Dict

class EFEPlanningAgent(nn.Module):
    """
    Active Inference agent using Expected Free Energy for planning.
    Implements MCTS-guided policy selection with epistemic exploration.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        planning_horizon: int = 5,
        num_simulations: int = 50,
        efe_precision: float = 1.0  # Inverse temperature for action selection
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.num_simulations = num_simulations
        self.efe_precision = efe_precision

        # Encoder: observations -> beliefs about states
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Output mean and log-var for state belief
        self.belief_mean = nn.Linear(hidden_dim, state_dim)
        self.belief_logvar = nn.Linear(hidden_dim, state_dim)

        # Transition model: p(x_t+1 | x_t, a_t)
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.trans_mean = nn.Linear(hidden_dim, state_dim)
        self.trans_logvar = nn.Linear(hidden_dim, state_dim)

        # Likelihood model: p(o | x)
        self.likelihood = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        # Preference model: log p~(o) - desired observations
        self.preference = nn.Parameter(torch.zeros(obs_dim))

        # Habitual network: fast policy for MCTS rollouts
        self.habitual = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def encode_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into belief state (posterior)."""
        h = self.encoder(obs)
        mean = self.belief_mean(h)
        logvar = self.belief_logvar(h)
        return mean, logvar

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state given current state and action."""
        # One-hot encode action if discrete
        if action.dim() == 1:
            action_oh = F.one_hot(action, self.action_dim).float()
        else:
            action_oh = action

        x = torch.cat([state, action_oh], dim=-1)
        h = self.transition(x)
        mean = self.trans_mean(h)
        logvar = self.trans_logvar(h)
        return mean, logvar

    def predict_observation(self, state: torch.Tensor) -> torch.Tensor:
        """Predict observation from state."""
        return self.likelihood(state)

    def compute_efe(
        self,
        prior_mean: torch.Tensor,
        prior_logvar: torch.Tensor,
        action: torch.Tensor,
        horizon: int = 1
    ) -> torch.Tensor:
        """
        Compute Expected Free Energy for a given action.

        G = -E[ln p~(o)] - E[D_KL[Q(x|o) || Q(x|pi)]]
          = pragmatic_value + epistemic_value
        """
        batch_size = prior_mean.shape[0]
        total_efe = torch.zeros(batch_size, device=prior_mean.device)

        current_mean = prior_mean
        current_logvar = prior_logvar

        for t in range(horizon):
            # Predict next state distribution (prior for next step)
            next_mean, next_logvar = self.predict_next_state(current_mean, action)

            # Sample expected observation
            pred_obs = self.predict_observation(next_mean)

            # PRAGMATIC VALUE: -E[ln p~(o)]
            # Higher preference = lower EFE (better)
            log_preference = F.log_softmax(self.preference, dim=-1)
            pragmatic = -torch.sum(pred_obs * log_preference, dim=-1)

            # EPISTEMIC VALUE: Expected information gain
            # Approximate as entropy of predicted state (uncertainty to resolve)
            # Full computation would need Q(x|o), here we use tractable bound
            prior_entropy = 0.5 * torch.sum(1 + next_logvar, dim=-1)

            # Epistemic value = negative entropy (want to reduce uncertainty)
            # But in EFE, it's negative, so we add entropy to encourage exploration
            epistemic = -prior_entropy

            total_efe += pragmatic + epistemic

            # Update for next timestep
            current_mean = next_mean
            current_logvar = next_logvar

            # Use habitual policy for subsequent actions
            if t < horizon - 1:
                action_logits = self.habitual(current_mean)
                action = torch.argmax(action_logits, dim=-1)

        return total_efe

    def compute_efe_full(
        self,
        obs: torch.Tensor,
        action: int,
        num_samples: int = 10
    ) -> float:
        """
        Compute EFE with full epistemic value using Monte Carlo sampling.
        More accurate but slower than compute_efe.
        """
        # Encode current observation
        post_mean, post_logvar = self.encode_observation(obs)

        # Sample states from posterior
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn(num_samples, self.state_dim, device=obs.device)
        states = post_mean + eps * std

        # Convert action to tensor
        action_tensor = torch.tensor([action], device=obs.device).expand(num_samples)

        # Predict next states
        next_mean, next_logvar = self.predict_next_state(states, action_tensor)

        # Sample next observations
        next_obs = self.predict_observation(next_mean)

        # PRAGMATIC: -E[ln p~(o)]
        log_pref = F.log_softmax(self.preference, dim=-1)
        pragmatic = -torch.mean(torch.sum(next_obs * log_pref, dim=-1))

        # EPISTEMIC: Expected KL between posterior and prior
        # KL[Q(x|o) || Q(x|pi)] = 0.5 * sum(var_ratio + mean_diff^2 - 1 - log_var_ratio)
        # Here we approximate using entropy difference

        prior_entropy = 0.5 * torch.mean(torch.sum(1 + next_logvar, dim=-1))

        # Approximate posterior entropy (would need observation model)
        # Using prior entropy as lower bound on epistemic value
        epistemic = -prior_entropy

        return (pragmatic + epistemic).item()

    def select_action_mcts(
        self,
        obs: torch.Tensor,
        temperature: float = 1.0
    ) -> int:
        """
        Select action using Monte Carlo Tree Search guided by EFE.
        """
        # Encode observation
        post_mean, post_logvar = self.encode_observation(obs)

        # Compute EFE for each action
        efe_values = []
        for a in range(self.action_dim):
            action = torch.tensor([a], device=obs.device)
            efe = self.compute_efe(
                post_mean, post_logvar, action,
                horizon=self.planning_horizon
            )
            efe_values.append(efe.item())

        # Convert to action probabilities (lower EFE = higher probability)
        efe_tensor = torch.tensor(efe_values, device=obs.device)
        action_probs = F.softmax(-self.efe_precision * efe_tensor / temperature, dim=-1)

        # Sample action
        action = Categorical(action_probs).sample().item()

        return action

    def forward(self, obs: torch.Tensor) -> int:
        """Main entry point for action selection."""
        return self.select_action_mcts(obs)


class MCTSNode:
    """Node in Monte Carlo Tree Search for active inference."""

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_logvar: torch.Tensor,
        parent=None,
        action=None
    ):
        self.state_mean = state_mean
        self.state_logvar = state_logvar
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.efe_sum = 0.0
        self.efe_squared_sum = 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def get_efe_mean(self) -> float:
        if self.visit_count == 0:
            return float('inf')
        return self.efe_sum / self.visit_count

    def get_efe_uncertainty(self) -> float:
        """Epistemic uncertainty about EFE estimate."""
        if self.visit_count < 2:
            return float('inf')
        mean = self.efe_sum / self.visit_count
        var = (self.efe_squared_sum / self.visit_count) - mean**2
        return np.sqrt(max(0, var) / self.visit_count)

    def update(self, efe: float):
        self.visit_count += 1
        self.efe_sum += efe
        self.efe_squared_sum += efe**2


class ActiveInferenceMCTS:
    """
    Monte Carlo Tree Search with Expected Free Energy.
    Connects EFE to UCB exploration bonus naturally.
    """

    def __init__(
        self,
        agent: EFEPlanningAgent,
        num_actions: int,
        exploration_weight: float = 1.0,
        num_simulations: int = 100
    ):
        self.agent = agent
        self.num_actions = num_actions
        self.exploration_weight = exploration_weight
        self.num_simulations = num_simulations

    def search(self, root_state_mean: torch.Tensor, root_state_logvar: torch.Tensor) -> int:
        """Run MCTS and return best action."""
        root = MCTSNode(root_state_mean, root_state_logvar)

        for _ in range(self.num_simulations):
            node = root
            path = []

            # Selection: navigate to leaf using UCB-like criterion
            while node.is_expanded():
                action = self._select_action(node)
                path.append((node, action))
                node = node.children[action]

            # Expansion: add children
            self._expand(node)

            # Simulation: compute EFE
            efe = self._simulate(node)

            # Backpropagation
            for parent, action in reversed(path):
                parent.children[action].update(efe)
                parent.update(efe)

        # Select best action
        best_action = min(
            range(self.num_actions),
            key=lambda a: root.children[a].get_efe_mean() if a in root.children else float('inf')
        )

        return best_action

    def _select_action(self, node: MCTSNode) -> int:
        """
        Select action using EFE-UCB criterion.
        Lower EFE is better, so we minimize.

        UCB for minimization:
        score = mean_efe - c * sqrt(ln(N) / n)
        """
        best_score = float('inf')
        best_action = 0

        log_total = np.log(node.visit_count + 1)

        for action, child in node.children.items():
            if child.visit_count == 0:
                return action  # Explore unvisited

            # EFE-UCB: minimize EFE with exploration bonus
            mean_efe = child.get_efe_mean()
            exploration = self.exploration_weight * np.sqrt(log_total / child.visit_count)

            # For minimization, subtract exploration term
            score = mean_efe - exploration

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def _expand(self, node: MCTSNode):
        """Expand node by adding all action children."""
        for action in range(self.num_actions):
            action_tensor = torch.tensor([action], device=node.state_mean.device)
            next_mean, next_logvar = self.agent.predict_next_state(
                node.state_mean, action_tensor
            )
            node.children[action] = MCTSNode(
                next_mean, next_logvar,
                parent=node, action=action
            )

    def _simulate(self, node: MCTSNode) -> float:
        """Compute EFE from this node."""
        # Use habitual policy for rollout
        state_mean = node.state_mean
        state_logvar = node.state_logvar
        total_efe = 0.0

        for _ in range(self.agent.planning_horizon):
            # Get habitual action
            action_logits = self.agent.habitual(state_mean)
            action = torch.argmax(action_logits, dim=-1)

            # Compute single-step EFE
            efe = self.agent.compute_efe(state_mean, state_logvar, action, horizon=1)
            total_efe += efe.item()

            # Transition
            state_mean, state_logvar = self.agent.predict_next_state(state_mean, action)

        return total_efe


# Training utilities
def compute_vfe_loss(
    agent: EFEPlanningAgent,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    action: torch.Tensor,
    beta: float = 1.0  # KL weight
) -> torch.Tensor:
    """
    Compute Variational Free Energy loss for training.
    VFE = -E[ln p(o|x)] + beta * KL[Q(x|o) || p(x)]
    """
    # Encode observations
    post_mean, post_logvar = agent.encode_observation(obs)

    # Sample latent state
    std = torch.exp(0.5 * post_logvar)
    eps = torch.randn_like(std)
    z = post_mean + eps * std

    # Reconstruction loss (negative log-likelihood)
    pred_obs = agent.predict_observation(z)
    recon_loss = F.mse_loss(pred_obs, obs, reduction='mean')

    # KL divergence from standard normal prior
    kl_loss = -0.5 * torch.mean(1 + post_logvar - post_mean.pow(2) - post_logvar.exp())

    # Transition prediction loss
    next_mean, next_logvar = agent.predict_next_state(z, action)
    next_post_mean, next_post_logvar = agent.encode_observation(next_obs)

    trans_loss = F.mse_loss(next_mean, next_post_mean.detach(), reduction='mean')

    return recon_loss + beta * kl_loss + trans_loss


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = EFEPlanningAgent(
        state_dim=32,
        obs_dim=64,
        action_dim=4,
        hidden_dim=128,
        planning_horizon=5,
        num_simulations=50
    )

    # Example observation
    obs = torch.randn(1, 64)

    # Select action using EFE planning
    action = agent.select_action_mcts(obs)
    print(f"Selected action: {action}")

    # With full MCTS
    post_mean, post_logvar = agent.encode_observation(obs)
    mcts = ActiveInferenceMCTS(agent, num_actions=4, num_simulations=100)
    action_mcts = mcts.search(post_mean, post_logvar)
    print(f"MCTS action: {action_mcts}")
```

### Performance Considerations

**Memory**:
- Store only sufficient statistics in MCTS nodes
- Use tensor views to avoid copying
- Batch state transitions when possible

**Speed**:
- Habitual network for fast rollouts
- Amortized inference via encoder
- GPU batching for EFE computation

**Numerical Stability**:
- Log-sum-exp for softmax over EFE
- Clamp log-variances
- Use numerically stable KL computation

Implementation reference: [Deep Active Inference MC GitHub](https://github.com/zfountas/deep-active-inference-mc)
- NeurIPS 2020 paper implementation
- Uses TensorFlow 2.0, but architecture translates to PyTorch
- Includes Animal-AI and dSprites environments

---

## Section 5: TRAIN STATION - EFE = UCB = Thompson Sampling

### The Deep Unification

**Coffee Cup = Donut**: These three exploration strategies are topologically equivalent!

```
EFE = -E[ln p~(o)] - E[KL[Q(x|o)||Q(x|pi)]]
         pragmatic      epistemic

UCB = Q(a) + c * sqrt(ln(N)/n(a))
       exploit    explore

Thompson = sample_from_posterior(reward)
              uncertainty-driven selection
```

### Why They're the Same

**1. EFE and UCB**

The epistemic value in EFE serves the same role as the UCB exploration bonus:
- Both increase with uncertainty
- Both decrease as that state/action is visited more
- Both provide principled exploration-exploitation balance

The mapping:
- `Q(a)` ~ `-E[ln p~(o)]` (expected value)
- `sqrt(ln(N)/n(a))` ~ `KL[Q(x|o)||Q(x|pi)]` (information gain)

**2. EFE and Thompson Sampling**

Thompson Sampling samples actions proportional to probability of being optimal.
EFE selects actions via softmax over negative EFE values:

```python
P(action) = softmax(-gamma * EFE(action))
          ~ P(action is optimal | posterior)
```

Both are posterior-weighted action selection!

**3. The Precision Parameter**

The `gamma` (inverse temperature) in EFE softmax corresponds to:
- UCB exploration coefficient `c`
- Thompson sampling posterior variance

High precision = exploit (greedy)
Low precision = explore (random)

### Mathematical Proof of Equivalence

For Gaussian posteriors over rewards:

```python
# Thompson Sampling
def thompson_select(posterior_means, posterior_vars):
    samples = [np.random.normal(m, np.sqrt(v)) for m, v in zip(posterior_means, posterior_vars)]
    return np.argmax(samples)

# Equivalent EFE formulation
def efe_select(pragmatic_values, epistemic_values, gamma=1.0):
    efe = -pragmatic_values - epistemic_values  # Both terms negative
    probs = softmax(-gamma * efe)
    return np.random.choice(len(efe), p=probs)
```

For Gaussian rewards:
- `pragmatic = -mean`
- `epistemic = -variance/2` (entropy of Gaussian)

This makes EFE selection equivalent to sampling from the posterior!

### Implications for Planning

**MCTS with EFE naturally does UCB-style exploration**:
- Visit counts provide uncertainty estimates
- EFE epistemic term provides exploration bonus
- No need for separate exploration parameter

**Why this matters**:
- Principled foundation for exploration
- Automatic tuning via precision learning
- Connects neuroscience (active inference) to AI (MCTS/bandits)

### Code: Demonstrating Equivalence

```python
import numpy as np
from scipy.special import softmax

def demonstrate_efe_ucb_equivalence():
    """Show that EFE selection matches UCB behavior."""
    np.random.seed(42)

    num_actions = 5
    num_steps = 1000

    # True reward means
    true_means = np.random.randn(num_actions)

    # Track selections and rewards
    ucb_counts = np.zeros(num_actions)
    ucb_sums = np.zeros(num_actions)

    efe_counts = np.zeros(num_actions)
    efe_sums = np.zeros(num_actions)

    for t in range(1, num_steps + 1):
        # UCB Selection
        ucb_values = np.zeros(num_actions)
        for a in range(num_actions):
            if ucb_counts[a] == 0:
                ucb_values[a] = float('inf')
            else:
                mean = ucb_sums[a] / ucb_counts[a]
                bonus = np.sqrt(2 * np.log(t) / ucb_counts[a])
                ucb_values[a] = mean + bonus

        ucb_action = np.argmax(ucb_values)
        ucb_reward = np.random.randn() + true_means[ucb_action]
        ucb_counts[ucb_action] += 1
        ucb_sums[ucb_action] += ucb_reward

        # EFE Selection
        pragmatic = np.zeros(num_actions)
        epistemic = np.zeros(num_actions)

        for a in range(num_actions):
            if efe_counts[a] == 0:
                pragmatic[a] = 0  # No information
                epistemic[a] = 10  # High uncertainty = high exploration
            else:
                mean = efe_sums[a] / efe_counts[a]
                var = 1.0 / efe_counts[a]  # Posterior variance
                pragmatic[a] = mean
                epistemic[a] = 0.5 * np.log(2 * np.pi * np.e * var)  # Entropy

        efe = -pragmatic - epistemic  # Lower is better
        probs = softmax(-efe)  # Higher prob for lower EFE
        efe_action = np.random.choice(num_actions, p=probs)

        efe_reward = np.random.randn() + true_means[efe_action]
        efe_counts[efe_action] += 1
        efe_sums[efe_action] += efe_reward

    print("Selection frequencies:")
    print(f"UCB: {ucb_counts / num_steps}")
    print(f"EFE: {efe_counts / num_steps}")
    print(f"Optimal action: {np.argmax(true_means)}")

    # Both should concentrate on optimal action
    return ucb_counts, efe_counts

if __name__ == "__main__":
    demonstrate_efe_ucb_equivalence()
```

### The Grand Unification

All these methods are doing the same thing: **balancing expected value against uncertainty**

| Method | Exploit Term | Explore Term | Selection |
|--------|--------------|--------------|-----------|
| EFE | -E[ln p~(o)] | -KL[post\|\|prior] | softmax(-EFE) |
| UCB | Q(a) | c*sqrt(ln(N)/n) | argmax |
| Thompson | posterior mean | posterior variance | sample & argmax |
| Info Gain | reward | mutual information | argmax |

**They're all the same algorithm with different parameterizations!**

---

## Section 6: ARR-COC-0-1 - Token Allocation as Planning

### The Connection: Relevance Realization as Active Inference

ARR-COC performs **token allocation** - deciding which image regions get more processing budget. This is directly analogous to **action selection** in active inference!

**Mapping**:
- **Actions** in active inference → **Token allocation decisions** in ARR-COC
- **Expected Free Energy** → **Expected Relevance Gain**
- **Epistemic value** → **Information gain from processing region**
- **Pragmatic value** → **Task-relevant value of region**

### Token Allocation as EFE Minimization

```python
def compute_token_efe(
    region_features: torch.Tensor,
    current_belief: torch.Tensor,
    task_embedding: torch.Tensor
) -> torch.Tensor:
    """
    Compute Expected Free Energy for token allocation.

    Regions with high EFE (to minimize) get fewer tokens.
    Regions with low EFE get more tokens.
    """
    # PRAGMATIC: Task relevance (goal-directed)
    task_relevance = torch.cosine_similarity(region_features, task_embedding)
    pragmatic = -task_relevance  # Lower = better (more relevant)

    # EPISTEMIC: Information gain from processing
    # Uncertainty about region content
    uncertainty = compute_belief_entropy(current_belief)
    epistemic = -uncertainty  # Lower = better (more uncertain = want to process)

    efe = pragmatic + epistemic
    return efe

def allocate_tokens_efe(
    image_regions: torch.Tensor,
    total_tokens: int,
    task_embedding: torch.Tensor,
    precision: float = 1.0
) -> torch.Tensor:
    """
    Allocate tokens to regions using EFE-based planning.
    """
    # Compute EFE for each region
    efe = compute_token_efe(image_regions, task_embedding)

    # Convert to allocation (lower EFE = more tokens)
    allocation_weights = F.softmax(-precision * efe, dim=-1)
    token_allocation = (allocation_weights * total_tokens).int()

    return token_allocation
```

### Hierarchical Planning for Multi-Scale Processing

Active inference naturally handles hierarchical planning through temporal abstraction. ARR-COC's pyramid structure maps perfectly:

```
Active Inference Hierarchy    ARR-COC Pyramid
─────────────────────────    ────────────────
High-level goals        ←→    Global image understanding
Mid-level policies      ←→    Region-level allocation
Low-level actions       ←→    Patch-level token assignment
```

Each level minimizes EFE at its timescale:
- Slow updates at top (overall task)
- Fast updates at bottom (local processing)

### Precision Weighting as Token Budget

The precision parameter in active inference controls confidence in predictions. In ARR-COC:

**High precision regions** = High confidence = Fewer tokens needed
**Low precision regions** = Low confidence = More tokens for disambiguation

This naturally implements adaptive computation!

```python
class PrecisionWeightedTokenAllocation(nn.Module):
    """
    Use learned precision to weight token allocation.
    Connects to active inference precision learning.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        # Predict precision (inverse variance) for each region
        self.precision_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()  # Precision must be positive
        )

    def forward(
        self,
        region_features: torch.Tensor,
        base_allocation: torch.Tensor
    ) -> torch.Tensor:
        # Predict precision for each region
        precision = self.precision_net(region_features)

        # High precision = confident = reduce allocation
        # Low precision = uncertain = increase allocation
        adjusted_allocation = base_allocation / (precision + 1e-6)

        # Normalize to maintain total budget
        adjusted_allocation = adjusted_allocation / adjusted_allocation.sum() * base_allocation.sum()

        return adjusted_allocation
```

### Implementation Sketch for ARR-COC

```python
class EFETokenRouter(nn.Module):
    """
    Route tokens using Expected Free Energy planning.
    Balances task relevance with information gain.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Models for EFE computation
        self.task_encoder = nn.Linear(config.text_dim, config.hidden_dim)
        self.region_encoder = nn.Linear(config.vision_dim, config.hidden_dim)

        # Epistemic value estimator (uncertainty)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        # Pragmatic value estimator (task relevance)
        self.relevance_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        self.precision = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        image_features: torch.Tensor,  # [B, N, D]
        task_embedding: torch.Tensor,  # [B, D_text]
        total_budget: int
    ) -> torch.Tensor:
        B, N, D = image_features.shape

        # Encode
        task_enc = self.task_encoder(task_embedding)  # [B, H]
        region_enc = self.region_encoder(image_features)  # [B, N, H]

        # Compute epistemic value (uncertainty per region)
        epistemic = self.uncertainty_head(region_enc).squeeze(-1)  # [B, N]

        # Compute pragmatic value (task relevance)
        task_expanded = task_enc.unsqueeze(1).expand(-1, N, -1)  # [B, N, H]
        combined = torch.cat([region_enc, task_expanded], dim=-1)  # [B, N, 2H]
        pragmatic = self.relevance_head(combined).squeeze(-1)  # [B, N]

        # Expected Free Energy (lower is better)
        efe = -pragmatic - epistemic

        # Convert to allocation via softmax
        allocation = F.softmax(-self.precision * efe, dim=-1) * total_budget

        return allocation.int()
```

### Performance Considerations for ARR-COC

**Computational Cost**: EFE computation adds overhead, but:
- Can be amortized via learned router
- Only computed once per forward pass
- Parallelizes well on GPU

**Memory**: Storing beliefs/uncertainties for all regions
- Use efficient representations (diagonal Gaussians)
- Share computation across pyramid levels

**Training**: Can learn EFE components via:
- Reconstruction loss (accuracy term in VFE)
- Task performance (pragmatic value)
- Prediction error (epistemic value)

---

## Sources

### Academic Papers

**Core EFE Theory**:
- [Millidge, Tschantz & Buckley (2021)](https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy) - "Whence the Expected Free Energy?" Neural Computation - Origin and decomposition analysis
- [Friston et al. (2015)](https://www.tandfonline.com/doi/full/10.1080/17588928.2015.1020053) - "Active Inference and Epistemic Value" - 920+ citations, foundational paper
- [Parr & Friston (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/) - "Generalised Free Energy and Active Inference" - Unified formulation

**MCTS and Planning**:
- [Fountas et al. (2020)](https://proceedings.neurips.cc/paper/2020/file/865dfbde8a344b44095495f3591f7407-Paper.pdf) - "Deep Active Inference Agents Using Monte-Carlo Methods" NeurIPS - 139 citations
- [Maisto et al. (2025)](https://www.sciencedirect.com/science/article/pii/S0925231224020903) - "Active Inference Tree Search in Large POMDPs" Neurocomputing
- [Dao et al. (2025)](https://direct.mit.edu/neco/article/37/12/2205/133239/Boosting-MCTS-With-Free-Energy-Minimization) - "Boosting MCTS with Free Energy Minimization" Neural Computation

**Variational Planning**:
- [de Vries (2025)](https://arxiv.org/abs/2504.14898) - "Expected Free Energy-based Planning as Variational Inference" arXiv
- [Friston et al. (2025)](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2025.1521963/full) - "From Pixels to Planning: Scale-Free Active Inference" Frontiers

### Code Repositories

- [Deep Active Inference MC](https://github.com/zfountas/deep-active-inference-mc) - TensorFlow implementation of NeurIPS 2020 paper
- [pymdp](https://github.com/infer-actively/pymdp) - Python package for discrete active inference

### Additional Resources

- [Active Inference Tutorial](https://medium.com/@solopchuk/tutorial-on-active-inference-30edcf50f5dc) - Comprehensive introduction
- [Active Inference Institute](https://www.youtube.com/c/ActiveInferenceInstitute) - Educational videos and livestreams

---

*Last Updated: 2025-11-23*
*Knowledge Domain: ML-Active-Inference / Planning*
