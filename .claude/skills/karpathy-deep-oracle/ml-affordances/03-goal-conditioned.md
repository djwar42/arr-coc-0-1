# Goal-Conditioned Learning: Goals as Affordances

## Overview

Goal-conditioned learning treats goals as explicit affordances - desired future states that guide action selection. This framing unifies goal-conditioned RL, expected free energy planning, and affordance-based action selection into a common framework where goals represent actionable possibilities.

**Core Insight**: A goal is an affordance - it specifies what future state can be achieved through action. Goal-conditioned policies learn which actions afford reaching which goals.

From [NeurIPS 2024 Paper on Goal-Conditioned On-Policy RL](https://openreview.net/forum?id=KP7EUORJYI) (accessed 2025-01-23):
- Goal-Conditioned RL (GCRL) learns policies that generalize across multiple goals
- Hindsight Experience Replay (HER) enables learning from failed trajectories
- Can handle both Markovian and non-Markovian reward structures

## Fundamentals of Goal-Conditioned Learning

### Goal Representations

Goals can be represented in multiple ways:

**State-Based Goals**: Desired state s_g
```python
import torch
import torch.nn as nn

class GoalConditionedPolicy(nn.Module):
    """Policy conditioned on goal state"""
    def __init__(self, state_dim, goal_dim, action_dim, hidden=256):
        super().__init__()

        # Concatenate state and goal
        self.policy = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, state, goal):
        # Policy π(a | s, g)
        x = torch.cat([state, goal], dim=-1)
        return self.policy(x)

# Usage
policy = GoalConditionedPolicy(state_dim=10, goal_dim=10, action_dim=4)
action = policy(state, goal)  # π(a | s, g)
```

**Feature-Based Goals**: Desired features φ(s_g)
```python
class FeatureGoalPolicy(nn.Module):
    """Policy conditioned on goal features"""
    def __init__(self, state_dim, feature_dim, action_dim):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Policy on feature space
        self.policy = nn.Sequential(
            nn.Linear(2 * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state, goal_features):
        # Encode current state
        state_features = self.state_encoder(state)

        # Concatenate in feature space
        x = torch.cat([state_features, goal_features], dim=-1)
        return self.policy(x)
```

**Distance-Based Goals**: Goal as target distance d_g
```python
class DistanceGoalPolicy(nn.Module):
    """Policy conditioned on goal distance"""
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # State + distance scalar
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state, goal_distance):
        # Distance can be scalar or vector
        if goal_distance.dim() == 1:
            goal_distance = goal_distance.unsqueeze(-1)
        x = torch.cat([state, goal_distance], dim=-1)
        return self.policy(x)
```

### Goal-Conditioned Value Functions

Value function that generalizes across goals:

```python
class GoalConditionedQFunction(nn.Module):
    """Q(s, a, g) - value of action a in state s for goal g"""
    def __init__(self, state_dim, action_dim, goal_dim, hidden=256):
        super().__init__()

        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.q_net(x)

class GoalConditionedValueFunction(nn.Module):
    """V(s, g) - value of state s for goal g"""
    def __init__(self, state_dim, goal_dim, hidden=256):
        super().__init__()

        self.v_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.v_net(x)
```

## Hindsight Experience Replay (HER)

### Core HER Algorithm

From [Andrychowicz et al. NeurIPS 2017](https://proceedings.neurips.cc/paper/7090-hindsight-experience-replay.pdf) (accessed 2025-01-23):

HER enables learning from failed trajectories by relabeling goals in hindsight. Key insight: Even if you didn't reach the intended goal, you reached *some* state - treat that as a valid goal.

```python
class HindsightReplayBuffer:
    """Replay buffer with hindsight goal relabeling"""
    def __init__(self, capacity, k_hindsight=4):
        self.capacity = capacity
        self.k_hindsight = k_hindsight  # Number of hindsight goals per trajectory
        self.buffer = []

    def store_trajectory(self, trajectory, original_goal):
        """
        Store trajectory with both original and hindsight goals

        trajectory: [(s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), ...]
        original_goal: g_original
        """
        T = len(trajectory)

        # 1. Store with original goal
        for t, (state, action, reward, next_state) in enumerate(trajectory):
            self.buffer.append({
                'state': state,
                'action': action,
                'goal': original_goal,
                'reward': reward,
                'next_state': next_state,
                'done': (reward > 0)  # Goal reached
            })

        # 2. Hindsight relabeling: use achieved states as goals
        for t in range(T):
            state, action, _, next_state = trajectory[t]

            # Sample k future states as hindsight goals
            for _ in range(self.k_hindsight):
                # Future strategy: sample from future states in trajectory
                future_t = np.random.randint(t, T)
                hindsight_goal = trajectory[future_t][3]  # s_{future}

                # Compute reward for this hindsight goal
                hindsight_reward = self._compute_reward(next_state, hindsight_goal)

                self.buffer.append({
                    'state': state,
                    'action': action,
                    'goal': hindsight_goal,
                    'reward': hindsight_reward,
                    'next_state': next_state,
                    'done': (hindsight_reward > 0)
                })

        # Maintain capacity
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def _compute_reward(self, state, goal, threshold=0.05):
        """Check if state matches goal (within threshold)"""
        distance = np.linalg.norm(state - goal)
        return 1.0 if distance < threshold else -1.0

    def sample(self, batch_size):
        """Sample batch for training"""
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        return {
            'states': torch.FloatTensor([b['state'] for b in batch]),
            'actions': torch.FloatTensor([b['action'] for b in batch]),
            'goals': torch.FloatTensor([b['goal'] for b in batch]),
            'rewards': torch.FloatTensor([b['reward'] for b in batch]),
            'next_states': torch.FloatTensor([b['next_state'] for b in batch]),
            'dones': torch.FloatTensor([b['done'] for b in batch])
        }
```

### HER Relabeling Strategies

Different strategies for selecting hindsight goals:

```python
class HERRelabelingStrategies:
    """Different strategies for hindsight goal relabeling"""

    @staticmethod
    def future_strategy(trajectory, t, k=4):
        """Sample k goals from future states in trajectory"""
        T = len(trajectory)
        goals = []
        for _ in range(k):
            future_t = np.random.randint(t, T)
            goal = trajectory[future_t]['state']
            goals.append(goal)
        return goals

    @staticmethod
    def final_strategy(trajectory, t, k=4):
        """Use final achieved state as goal"""
        final_state = trajectory[-1]['state']
        return [final_state] * k

    @staticmethod
    def episode_strategy(trajectory, t, k=4):
        """Sample k random states from entire trajectory"""
        T = len(trajectory)
        goals = []
        for _ in range(k):
            random_t = np.random.randint(0, T)
            goal = trajectory[random_t]['state']
            goals.append(goal)
        return goals

    @staticmethod
    def random_strategy(trajectory, t, k=4, state_dim=10):
        """Sample k completely random goals"""
        return [np.random.randn(state_dim) for _ in range(k)]

# Usage comparison
def compare_her_strategies(trajectory):
    """Compare different HER relabeling strategies"""
    t = len(trajectory) // 2  # Middle of trajectory

    future_goals = HERRelabelingStrategies.future_strategy(trajectory, t)
    final_goals = HERRelabelingStrategies.final_strategy(trajectory, t)
    episode_goals = HERRelabelingStrategies.episode_strategy(trajectory, t)

    print(f"Future strategy: {len(future_goals)} goals from future states")
    print(f"Final strategy: {len(final_goals)} goals (all final state)")
    print(f"Episode strategy: {len(episode_goals)} goals from any state")

    # Future strategy generally works best
    return future_goals
```

### Complete HER Training Loop

```python
def train_goal_conditioned_agent_with_her(
    env,
    policy,
    q_function,
    num_epochs=100,
    num_episodes=50,
    batch_size=128,
    lr=3e-4
):
    """Train goal-conditioned agent with HER"""

    # Optimizers
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    q_optimizer = torch.optim.Adam(q_function.parameters(), lr=lr)

    # HER replay buffer
    replay_buffer = HindsightReplayBuffer(capacity=100000, k_hindsight=4)

    for epoch in range(num_epochs):
        epoch_rewards = []

        # Collect episodes
        for episode in range(num_episodes):
            # Sample goal
            goal = env.sample_goal()

            # Rollout trajectory
            trajectory = []
            state = env.reset()
            episode_reward = 0

            for step in range(env.max_steps):
                # Select action π(a | s, g)
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    goal_t = torch.FloatTensor(goal).unsqueeze(0)
                    action = policy(state_t, goal_t).squeeze(0).numpy()

                # Environment step
                next_state, reward, done, _ = env.step(action)

                trajectory.append((state, action, reward, next_state))
                episode_reward += reward
                state = next_state

                if done:
                    break

            # Store trajectory with HER
            replay_buffer.store_trajectory(trajectory, goal)
            epoch_rewards.append(episode_reward)

        # Training updates
        for _ in range(40):  # 40 gradient steps per epoch
            # Sample batch
            batch = replay_buffer.sample(batch_size)

            # Update Q-function
            with torch.no_grad():
                next_actions = policy(batch['next_states'], batch['goals'])
                target_q = batch['rewards'] + 0.98 * (1 - batch['dones']) * \
                           q_function(batch['next_states'], next_actions, batch['goals'])

            current_q = q_function(batch['states'], batch['actions'], batch['goals'])
            q_loss = nn.MSELoss()(current_q.squeeze(), target_q.squeeze())

            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # Update policy
            actions = policy(batch['states'], batch['goals'])
            policy_loss = -q_function(batch['states'], actions, batch['goals']).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

        print(f"Epoch {epoch}: Avg Reward = {np.mean(epoch_rewards):.3f}")

    return policy, q_function
```

## Goal Representations in Practice

### Universal Value Function Approximators (UVFA)

Learn value function that generalizes across goals:

```python
class UniversalValueFunction(nn.Module):
    """UVFA: V(s, g) for any state-goal pair"""
    def __init__(self, state_dim, goal_dim, hidden=256):
        super().__init__()

        # Shared encoding
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Value head
        self.value_head = nn.Linear(hidden, 1)

        # Goal-reaching probability head
        self.success_head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, state, goal):
        features = self.encoder(torch.cat([state, goal], dim=-1))
        value = self.value_head(features)
        success_prob = self.success_head(features)
        return value, success_prob

# Multi-head learning
def train_uvfa(uvfa, replay_buffer, batch_size=128):
    """Train UVFA with both value and success prediction"""
    batch = replay_buffer.sample(batch_size)

    # Value prediction
    values, success_probs = uvfa(batch['states'], batch['goals'])

    # Target value
    with torch.no_grad():
        next_values, _ = uvfa(batch['next_states'], batch['goals'])
        target_values = batch['rewards'] + 0.98 * (1 - batch['dones']) * next_values

    value_loss = nn.MSELoss()(values.squeeze(), target_values.squeeze())

    # Success prediction (auxiliary task)
    success_labels = (batch['rewards'] > 0).float()
    success_loss = nn.BCELoss()(success_probs.squeeze(), success_labels)

    # Combined loss
    total_loss = value_loss + 0.1 * success_loss
    return total_loss
```

### Goal Embeddings

Learn latent goal representations:

```python
class GoalEmbeddingNetwork(nn.Module):
    """Learn latent goal embeddings"""
    def __init__(self, state_dim, embedding_dim=32):
        super().__init__()

        # Encoder: state → embedding
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        # Decoder: embedding → state
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

    def encode_goal(self, goal_state):
        """Map goal state to latent embedding"""
        return self.encoder(goal_state)

    def decode_goal(self, goal_embedding):
        """Reconstruct goal state from embedding"""
        return self.decoder(goal_embedding)

    def forward(self, goal_state):
        embedding = self.encode_goal(goal_state)
        reconstruction = self.decode_goal(embedding)
        return reconstruction

# Train goal embeddings
def train_goal_embeddings(goal_encoder, goal_states, epochs=100, lr=1e-3):
    """Learn compressed goal representations"""
    optimizer = torch.optim.Adam(goal_encoder.parameters(), lr=lr)

    for epoch in range(epochs):
        # Autoencoder loss
        reconstructions = goal_encoder(goal_states)
        loss = nn.MSELoss()(reconstructions, goal_states)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Reconstruction Loss = {loss.item():.4f}")

    return goal_encoder

# Use in policy
class GoalEmbeddingPolicy(nn.Module):
    """Policy using learned goal embeddings"""
    def __init__(self, state_dim, embedding_dim, action_dim):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state, goal_embedding):
        x = torch.cat([state, goal_embedding], dim=-1)
        return self.policy(x)
```

## Code: Complete Goal-Conditioned Agent

Full implementation with HER and goal embeddings:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class GoalConditionedActor(nn.Module):
    """Goal-conditioned policy network"""
    def __init__(self, state_dim, goal_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, state, goal):
        return self.net(torch.cat([state, goal], dim=-1))

class GoalConditionedCritic(nn.Module):
    """Goal-conditioned Q-function"""
    def __init__(self, state_dim, goal_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, goal, action):
        return self.net(torch.cat([state, goal, action], dim=-1))

class HERAgent:
    """Complete goal-conditioned agent with HER"""
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        her_k=4
    ):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.her_k = her_k

        # Networks
        self.actor = GoalConditionedActor(state_dim, goal_dim, action_dim)
        self.actor_target = GoalConditionedActor(state_dim, goal_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = GoalConditionedCritic(state_dim, goal_dim, action_dim)
        self.critic_target = GoalConditionedCritic(state_dim, goal_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)

    def select_action(self, state, goal, noise=0.1):
        """Select action with exploration noise"""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        goal_t = torch.FloatTensor(goal).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_t, goal_t).squeeze(0).numpy()

        # Add exploration noise
        action += noise * np.random.randn(self.action_dim)
        return np.clip(action, -1.0, 1.0)

    def store_episode(self, episode, original_goal):
        """Store episode with HER relabeling"""
        T = len(episode)

        # Store original goal transitions
        for transition in episode:
            self.buffer.append({
                'state': transition['state'],
                'action': transition['action'],
                'goal': original_goal,
                'reward': transition['reward'],
                'next_state': transition['next_state'],
                'done': transition['done']
            })

        # HER: Relabel with future achieved states
        for t in range(T):
            for _ in range(self.her_k):
                # Sample future state as hindsight goal
                future_t = np.random.randint(t, T)
                hindsight_goal = episode[future_t]['next_state']

                # Recompute reward for hindsight goal
                achieved = episode[t]['next_state']
                distance = np.linalg.norm(achieved - hindsight_goal)
                hindsight_reward = 0.0 if distance < 0.05 else -1.0
                hindsight_done = (hindsight_reward == 0.0)

                self.buffer.append({
                    'state': episode[t]['state'],
                    'action': episode[t]['action'],
                    'goal': hindsight_goal,
                    'reward': hindsight_reward,
                    'next_state': episode[t]['next_state'],
                    'done': hindsight_done
                })

    def train_step(self):
        """Single training step"""
        if len(self.buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        states = torch.FloatTensor([b['state'] for b in batch])
        actions = torch.FloatTensor([b['action'] for b in batch])
        goals = torch.FloatTensor([b['goal'] for b in batch])
        rewards = torch.FloatTensor([b['reward'] for b in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([b['next_state'] for b in batch])
        dones = torch.FloatTensor([b['done'] for b in batch]).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states, goals)
            target_q = self.critic_target(next_states, goals, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, goals, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        policy_actions = self.actor(states, goals)
        actor_loss = -self.critic(states, goals, policy_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

    def _soft_update(self, source, target):
        """Soft update target network"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

# Training loop
def train_her_agent(env, agent, num_epochs=100, episodes_per_epoch=50):
    """Train HER agent"""
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_success = []

        for _ in range(episodes_per_epoch):
            # Sample goal
            goal = env.sample_goal()

            # Rollout episode
            episode = []
            state = env.reset()
            episode_reward = 0

            for step in range(env.max_steps):
                action = agent.select_action(state, goal, noise=0.1)
                next_state, reward, done, info = env.step(action)

                episode.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })

                episode_reward += reward
                state = next_state

                if done:
                    break

            # Store with HER
            agent.store_episode(episode, goal)

            epoch_rewards.append(episode_reward)
            epoch_success.append(float(done))

        # Training updates
        for _ in range(40):
            losses = agent.train_step()

        print(f"Epoch {epoch}: Reward={np.mean(epoch_rewards):.2f}, "
              f"Success={np.mean(epoch_success):.2%}")
```

## TRAIN STATION: Goal = Affordance = Expected Free Energy = Planning

**The Coffee Cup = Donut Unification:**

Goals, affordances, expected free energy, and planning are topologically equivalent - different framings of "what future states can I reach?"

### Equivalence 1: Goal = Affordance

**Gibson's Affordances**: Objects afford actions
- Cup affords grasping
- Chair affords sitting

**Goal-Conditioned RL**: Goals afford policies
- Goal g affords policy π(a|s,g)
- Reaching goal g is an affordance of the current state

```python
# Affordance detection = Goal reachability
def compute_goal_affordances(state, goals, policy, horizon=10):
    """Which goals are afforded from this state?"""
    affordances = {}

    for goal in goals:
        # Can we reach this goal?
        reachability = simulate_goal_reaching(state, goal, policy, horizon)
        affordances[goal] = reachability

    return affordances

# This IS affordance detection!
# Goal g is afforded if π(a|s,g) can reach g
```

### Equivalence 2: Goal = Expected Free Energy

**Active Inference**: Minimize expected free energy
- Epistemic value: Reduce uncertainty (information gain)
- Pragmatic value: Achieve preferred outcomes (goal-reaching)

**Goal-Conditioned Learning**: Maximize goal-reaching probability
- Same objective: Q(s,a,g) = expected return for goal g
- EFE = -Q(s,a,g) (minimize energy = maximize value)

```python
# Expected free energy = Goal-conditioned value
class EFEAsGoalValue(nn.Module):
    """EFE framing of goal-conditioned value"""
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()

        # Pragmatic value: Q(s, a, g)
        self.pragmatic_value = GoalConditionedCritic(
            state_dim, goal_dim, action_dim
        )

        # Epistemic value: Information gain about goal
        self.epistemic_value = nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def expected_free_energy(self, state, action, goal):
        # EFE = -(pragmatic + epistemic)
        pragmatic = self.pragmatic_value(state, goal, action)
        epistemic = self.epistemic_value(
            torch.cat([state, goal, action], dim=-1)
        )

        # Minimize EFE = maximize value
        efe = -(pragmatic + 0.1 * epistemic)
        return efe

    def select_action(self, state, goal, action_candidates):
        """Select action minimizing EFE"""
        efes = [
            self.expected_free_energy(state, a, goal)
            for a in action_candidates
        ]
        return action_candidates[torch.argmin(torch.stack(efes))]
```

### Equivalence 3: Goal-Conditioned = Planning

**Model-Based Planning**: Search for action sequence reaching goal
- Forward search: Which states can I reach?
- Backward search: How do I reach goal state?

**Goal-Conditioned Policy**: Amortized planning
- Policy π(a|s,g) implicitly plans path to g
- HER = learning from all possible "plans" (trajectories)

```python
# Goal-conditioned policy = Learned planner
class GoalConditionedPlanner:
    """Goal-conditioned policy IS a planner"""
    def __init__(self, policy, model):
        self.policy = policy  # π(a|s,g)
        self.model = model    # T(s'|s,a)

    def explicit_planning(self, state, goal, horizon=10):
        """Traditional planning: search action sequences"""
        best_actions = None
        best_cost = float('inf')

        # Search over action sequences
        for _ in range(100):  # Monte Carlo tree search
            actions = []
            s = state
            cost = 0

            for t in range(horizon):
                a = np.random.randn(self.action_dim)
                s_next = self.model.predict(s, a)
                cost += np.linalg.norm(s_next - goal)
                s = s_next
                actions.append(a)

            if cost < best_cost:
                best_cost = cost
                best_actions = actions

        return best_actions[0]  # First action

    def implicit_planning(self, state, goal):
        """Goal-conditioned policy: amortized planning"""
        # This IS planning - just learned/amortized!
        action = self.policy(state, goal)
        return action

    # Same result, different computation:
    # explicit_planning: search at inference time
    # implicit_planning: search during training (HER)
```

### The Unified Framework

```python
class UnifiedGoalAffordancePlanner:
    """
    Goals = Affordances = EFE = Planning

    All answer: "What future states can I reach from here?"
    """
    def __init__(self, state_dim, goal_dim, action_dim):
        # One network, four interpretations
        self.network = GoalConditionedActor(state_dim, goal_dim, action_dim)

    def as_goal_conditioned_policy(self, state, goal):
        """Interpretation 1: Goal-conditioned RL"""
        return self.network(state, goal)

    def as_affordance_detector(self, state, goal):
        """Interpretation 2: Affordance - can I reach goal?"""
        action = self.network(state, goal)
        # Affordance = reachability
        return action  # If network trained, goal is afforded

    def as_efe_minimizer(self, state, goal):
        """Interpretation 3: Expected free energy"""
        action = self.network(state, goal)
        # Action minimizes EFE for reaching goal
        return action

    def as_planner(self, state, goal):
        """Interpretation 4: Amortized planning"""
        action = self.network(state, goal)
        # Action is first step of implicit plan to goal
        return action

    # These are THE SAME FUNCTION!
    # Different names, same computation
    # Goal = Affordance = EFE = Plan
```

**The Topological Insight:**

```python
# Coffee cup = donut (topologically)
# Goal = affordance = EFE = plan (functionally)

def show_equivalence():
    """All compute the same thing"""

    # Problem: Given state s and goal g, what action a?

    # Answer 1 (GCRL): a = argmax_a Q(s,a,g)
    # Answer 2 (Affordance): a that affords reaching g
    # Answer 3 (Active Inference): a = argmin_a EFE(s,a,g)
    # Answer 4 (Planning): a = first action in plan to g

    # These are equivalent!
    # All solve: "How do I get from s to g?"
```

## ARR-COC-0-1 Connection: Goal-Directed Relevance (10%)

### Goals as Relevance Targets

In ARR-COC, relevance scores could be goal-conditioned:

```python
class GoalConditionedRelevance(nn.Module):
    """Compute relevance conditioned on processing goal"""
    def __init__(self, token_dim, goal_dim, hidden=256):
        super().__init__()

        # Goal-conditioned relevance scorer
        self.relevance = nn.Sequential(
            nn.Linear(token_dim + goal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens, goal):
        """
        tokens: [batch, seq_len, token_dim]
        goal: [batch, goal_dim] - task objective

        Returns: relevance [batch, seq_len]
        """
        batch_size, seq_len, _ = tokens.shape

        # Expand goal for each token
        goal_expanded = goal.unsqueeze(1).expand(-1, seq_len, -1)

        # Compute goal-conditioned relevance
        token_goal = torch.cat([tokens, goal_expanded], dim=-1)
        relevance = self.relevance(token_goal).squeeze(-1)

        return relevance

# Different goals = different relevance patterns
qa_goal = encode_goal("answer question")
summarization_goal = encode_goal("summarize document")

# Same tokens, different relevance based on goal
relevance_qa = model(tokens, qa_goal)         # High for answer-related
relevance_summary = model(tokens, summarization_goal)  # High for key points
```

### Hindsight Relevance Replay

Apply HER principle to relevance learning:

```python
class HindsightRelevanceBuffer:
    """Learn relevance from both intended and achieved goals"""
    def __init__(self):
        self.buffer = []

    def store_with_hindsight(self, tokens, intended_goal, output):
        """
        Store relevance patterns with both:
        1. Intended goal (what we wanted)
        2. Achieved output (what we got) - hindsight goal
        """
        # Original: relevance for intended goal
        relevance_intended = compute_relevance(tokens, intended_goal)
        self.buffer.append((tokens, intended_goal, relevance_intended))

        # Hindsight: use actual output as goal
        achieved_goal = encode_output_as_goal(output)
        relevance_achieved = compute_relevance(tokens, achieved_goal)
        self.buffer.append((tokens, achieved_goal, relevance_achieved))

        # Learn: "tokens that led to output were relevant FOR that output"
        # Even if output wasn't what we wanted!
```

### Goal-Conditioned Token Allocation

Allocate tokens based on goal:

```python
def goal_conditioned_allocation(
    tokens,
    processing_goal,
    budget,
    relevance_model
):
    """
    Allocate compute budget based on goal

    Different goals → different allocation patterns
    """
    # Compute goal-conditioned relevance
    relevance = relevance_model(tokens, processing_goal)

    # Allocate budget to most goal-relevant tokens
    top_k = min(budget, len(tokens))
    selected_indices = torch.topk(relevance, top_k).indices

    # Process only goal-relevant tokens
    selected_tokens = tokens[selected_indices]
    return selected_tokens

# Example: Multiple goals, different allocations
qa_tokens = goal_conditioned_allocation(
    all_tokens,
    goal="answer_question",
    budget=100,
    relevance_model
)  # Selects answer-relevant tokens

summary_tokens = goal_conditioned_allocation(
    all_tokens,
    goal="summarize",
    budget=100,
    relevance_model
)  # Selects key-point tokens

# Same input, different processing based on goal!
```

## Performance Notes

### Computational Complexity

**HER Storage Cost:**
- Standard buffer: O(T) per episode (T = episode length)
- HER with k=4: O(T + 4T) = O(5T) per episode
- 5× storage but dramatically improved sample efficiency

**Training Speedup:**
```python
# Measure HER speedup
def compare_her_efficiency():
    """HER achieves same performance with 10× fewer episodes"""

    # Without HER: 10,000 episodes to converge
    episodes_no_her = 10000

    # With HER (k=4): 1,000 episodes to converge
    episodes_her = 1000

    # But 5× more transitions per episode
    total_transitions_no_her = episodes_no_her * avg_episode_length
    total_transitions_her = episodes_her * avg_episode_length * 5

    # Still 2× more sample efficient!
    speedup = total_transitions_no_her / total_transitions_her
    print(f"HER speedup: {speedup:.1f}×")
```

### Memory Requirements

**Goal-Conditioned Networks:**
```python
# Standard policy: π(a|s)
params_standard = state_dim * hidden + hidden * action_dim

# Goal-conditioned: π(a|s,g)
params_goal_conditioned = (state_dim + goal_dim) * hidden + hidden * action_dim

# Overhead
overhead = (state_dim + goal_dim) / state_dim
print(f"Parameter overhead: {overhead:.1f}×")
# Typically 1.5-2× parameters for goal dimension = state dimension
```

### GPU Optimization

```python
# Batch goal-conditioned inference
@torch.jit.script
def batch_goal_conditioned_forward(
    states: torch.Tensor,    # [batch, state_dim]
    goals: torch.Tensor,     # [batch, goal_dim]
    policy_weights: torch.Tensor
) -> torch.Tensor:
    """Optimized batch processing"""
    # Fused concatenation + matmul
    sg = torch.cat([states, goals], dim=1)
    return torch.matmul(sg, policy_weights)

# Faster than sequential forward passes
```

## Sources

**Source Documents:**
- None (external research only)

**Web Research:**
- [NeurIPS 2024: Goal-Conditioned On-Policy RL](https://openreview.net/forum?id=KP7EUORJYI) - GCPO framework for Markovian and non-Markovian rewards (accessed 2025-01-23)
- [Andrychowicz et al. NeurIPS 2017: Hindsight Experience Replay](https://proceedings.neurips.cc/paper/7090-hindsight-experience-replay.pdf) - Original HER paper (accessed 2025-01-23)
- [GOOSE: Goal-Conditioned RL for Safety-Critical Scenarios](https://arxiv.org/abs/2406.03870) - arXiv:2406.03870, application to autonomous driving (accessed 2025-01-23)

**Additional References:**
- Universal Value Function Approximators (Schaul et al. 2015)
- Goal-conditioned supervised learning
- Automatic curriculum learning via goal sampling
