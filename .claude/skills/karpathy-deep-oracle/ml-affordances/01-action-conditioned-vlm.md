# Action-Conditioned Vision-Language Models for Robotics

**ML-HEAVY IMPLEMENTATION GUIDE**

Action-conditioned Vision-Language Models (VLAs) represent a breakthrough in embodied AI by unifying perception, language understanding, and action prediction into a single end-to-end model. These models extend pre-trained VLMs with robotic action outputs.

---

## 1. Core Architecture: From VLM to VLA

### 1.1 The Vision-Language-Action Paradigm

**Key Innovation**: Treat robot actions as "another language" that can be tokenized and learned alongside natural language.

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class VisionLanguageActionModel(nn.Module):
    """
    Action-conditioned VLM architecture

    Flow: Image → VLM Encoder → Action Tokens + Text Tokens → Actions
    """
    def __init__(
        self,
        vlm_backbone: str = "google/paligemma-3b-pt-224",
        action_dim: int = 7,  # e.g., (x, y, z, roll, pitch, yaw, gripper)
        action_vocab_size: int = 256,  # token bins per action dimension
        max_horizon: int = 1  # predict N-step actions
    ):
        super().__init__()

        # Pre-trained VLM backbone (frozen or fine-tuned)
        self.vlm = AutoModel.from_pretrained(vlm_backbone)
        self.hidden_dim = self.vlm.config.hidden_size

        # Action tokenization parameters
        self.action_dim = action_dim
        self.action_vocab_size = action_vocab_size
        self.max_horizon = max_horizon

        # Action head: predict discrete action tokens
        self.action_head = nn.Linear(
            self.hidden_dim,
            action_dim * action_vocab_size * max_horizon
        )

        # Action token embeddings (for autoregressive generation)
        self.action_token_embed = nn.Embedding(
            action_vocab_size,
            self.hidden_dim
        )

    def forward(
        self,
        pixel_values: torch.Tensor,  # (B, 3, H, W)
        input_ids: torch.Tensor,     # (B, seq_len) - text tokens
        attention_mask: torch.Tensor,
        past_actions: torch.Tensor = None  # (B, action_dim) - optional conditioning
    ):
        """
        Forward pass: image + text + (optional) past actions → action tokens
        """
        # VLM encoding
        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Extract final hidden state (B, seq_len, hidden_dim)
        hidden_states = outputs.last_hidden_state

        # Pool to single vector (use [CLS] token or mean pooling)
        pooled = hidden_states[:, 0, :]  # (B, hidden_dim)

        # Optionally condition on past actions
        if past_actions is not None:
            # Discretize past actions to tokens
            past_action_tokens = self.discretize_actions(past_actions)
            past_embed = self.action_token_embed(past_action_tokens).mean(dim=1)
            pooled = pooled + past_embed

        # Predict action tokens (B, action_dim * vocab_size * horizon)
        action_logits = self.action_head(pooled)

        # Reshape to (B, horizon, action_dim, vocab_size)
        action_logits = action_logits.view(
            -1, self.max_horizon, self.action_dim, self.action_vocab_size
        )

        return action_logits

    def discretize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous actions to discrete tokens

        Args:
            actions: (B, action_dim) in range [-1, 1]

        Returns:
            tokens: (B, action_dim) in range [0, vocab_size-1]
        """
        # Normalize to [0, 1]
        normalized = (actions + 1.0) / 2.0

        # Quantize to bins
        tokens = (normalized * (self.action_vocab_size - 1)).long()
        tokens = torch.clamp(tokens, 0, self.action_vocab_size - 1)

        return tokens

    def undiscretize_actions(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete tokens back to continuous actions

        Args:
            tokens: (B, action_dim) in range [0, vocab_size-1]

        Returns:
            actions: (B, action_dim) in range [-1, 1]
        """
        # Convert to [0, 1]
        normalized = tokens.float() / (self.action_vocab_size - 1)

        # Convert to [-1, 1]
        actions = normalized * 2.0 - 1.0

        return actions

    @torch.no_grad()
    def predict_action(
        self,
        pixel_values: torch.Tensor,
        text_prompt: str,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Inference: predict continuous action from image + text
        """
        self.eval()

        # Tokenize text
        inputs = tokenizer(
            text_prompt,
            return_tensors="pt",
            padding=True
        ).to(device)

        # Forward pass
        action_logits = self.forward(
            pixel_values=pixel_values.to(device),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        # Get most likely tokens (greedy decoding)
        # (B, horizon, action_dim)
        action_tokens = action_logits.argmax(dim=-1)

        # Convert to continuous actions
        # Take first timestep if multi-horizon
        action_tokens_t0 = action_tokens[:, 0, :]  # (B, action_dim)
        continuous_actions = self.undiscretize_actions(action_tokens_t0)

        return continuous_actions


# Example usage
if __name__ == "__main__":
    model = VisionLanguageActionModel(
        action_dim=7,
        action_vocab_size=256,
        max_horizon=1
    )

    # Dummy inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text_ids = torch.randint(0, 1000, (batch_size, 20))
    mask = torch.ones(batch_size, 20)

    # Forward pass
    action_logits = model(images, text_ids, mask)
    print(f"Action logits shape: {action_logits.shape}")
    # Output: (4, 1, 7, 256) = (batch, horizon, action_dim, vocab_size)
```

**Performance Notes**:
- **Action discretization**: 256 bins provides ~0.4% resolution per dimension
- **Inference latency**: ~50-100ms on GPU (depends on VLM size)
- **Memory**: 3B param VLM ≈ 12GB VRAM, 55B ≈ 220GB VRAM

---

## 2. RT-2: Google's Action-Conditioned Transformer

### 2.1 RT-2 Architecture Details

From the RT-2 paper (2023), Google DeepMind showed VLMs can be fine-tuned for robotic control:

**Key Design Choices**:
1. **Co-fine-tuning**: Train on web data + robot data simultaneously
2. **Action tokenization**: Represent actions as text strings (e.g., "1 128 91 241 5 101 127 217")
3. **Backbones tested**: PaLM-E (12B), PaLI-X (55B)

```python
class RT2Model(nn.Module):
    """
    RT-2 style architecture: VLM → action tokens

    Based on: https://robotics-transformer2.github.io/
    """
    def __init__(
        self,
        vlm_name: str = "google/paligemma-3b-pt-224",
        action_dim: int = 7,
        action_bins: int = 256,
        use_diffusion: bool = False  # RT-2-X uses diffusion
    ):
        super().__init__()

        from transformers import AutoModelForCausalLM

        # Load pre-trained VLM
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_name)

        self.action_dim = action_dim
        self.action_bins = action_bins
        self.use_diffusion = use_diffusion

        # Action prediction head
        if not use_diffusion:
            # Discrete action tokens (original RT-2)
            self.action_proj = nn.Linear(
                self.vlm.config.hidden_size,
                action_dim * action_bins
            )
        else:
            # Diffusion-based actions (RT-2-X)
            from diffusers import UNet2DModel
            self.diffusion_net = UNet2DModel(
                sample_size=action_dim,
                in_channels=1,
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(128, 256, 512),
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D"
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D"
                )
            )

    def forward(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        # VLM forward pass
        outputs = self.vlm(
            input_ids=text_tokens,
            attention_mask=attention_mask,
            # Note: actual implementation interleaves image tokens
            # This is simplified for clarity
        )

        hidden = outputs.last_hidden_state[:, -1, :]  # Last token

        if not self.use_diffusion:
            # Discrete action prediction
            action_logits = self.action_proj(hidden)
            action_logits = action_logits.view(
                -1, self.action_dim, self.action_bins
            )
            return action_logits
        else:
            # Diffusion-based continuous actions
            # (simplified - real RT-2-X is more complex)
            return self.diffusion_net(hidden.unsqueeze(-1).unsqueeze(-1))
```

**RT-2 Results** (from paper):
- **Emergent capabilities**: 3x improvement on unseen objects
- **Symbol understanding**: Can follow "move to number 6" or "pick the extinct animal"
- **Chain-of-thought**: Can reason before acting
- **Generalization**: 2x better than RT-1 on novel scenarios

---

## 3. Training Action-Conditioned VLMs

### 3.1 Co-Fine-Tuning Recipe

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

class RobotDataset(Dataset):
    """
    Dataset mixing robot trajectories + web vision-language data
    """
    def __init__(
        self,
        robot_data_path: str,
        web_data_path: str,
        mix_ratio: float = 0.5  # 50% robot, 50% web
    ):
        self.robot_trajectories = self.load_robot_data(robot_data_path)
        self.web_examples = self.load_web_data(web_data_path)
        self.mix_ratio = mix_ratio

    def __getitem__(self, idx):
        # Sample from robot data or web data
        if torch.rand(1).item() < self.mix_ratio:
            return self.get_robot_example(idx)
        else:
            return self.get_web_example(idx)

    def get_robot_example(self, idx):
        """
        Robot trajectory: (image, language, action)
        """
        traj = self.robot_trajectories[idx % len(self.robot_trajectories)]

        return {
            "image": traj["observation"]["image"],
            "text": traj["instruction"],
            "action": traj["action"],  # (7,) continuous
            "is_robot_data": True
        }

    def get_web_example(self, idx):
        """
        Web VQA: (image, question, answer) - no action
        """
        example = self.web_examples[idx % len(self.web_examples)]

        return {
            "image": example["image"],
            "text": example["question"],
            "answer_text": example["answer"],
            "is_robot_data": False
        }


def train_vla_model(
    model: VisionLanguageActionModel,
    robot_data_path: str,
    web_data_path: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda"
):
    """
    Co-fine-tune VLM on robot + web data
    """
    # Setup
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Dataset
    dataset = RobotDataset(robot_data_path, web_data_path, mix_ratio=0.5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Scheduler
    num_training_steps = len(loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Loss functions
    action_loss_fn = nn.CrossEntropyLoss()  # For discrete actions
    text_loss_fn = nn.CrossEntropyLoss()    # For VQA responses

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            # Get batch data
            images = batch["image"].to(device)
            is_robot = batch["is_robot_data"]

            if is_robot.any():
                # Robot data: predict actions
                robot_indices = is_robot.nonzero(as_tuple=True)[0]
                robot_images = images[robot_indices]
                robot_texts = [batch["text"][i] for i in robot_indices]
                robot_actions = batch["action"][robot_indices].to(device)

                # Tokenize text
                text_inputs = tokenizer(
                    robot_texts,
                    return_tensors="pt",
                    padding=True
                ).to(device)

                # Forward pass
                action_logits = model(
                    pixel_values=robot_images,
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"]
                )

                # Discretize ground truth actions
                action_tokens = model.discretize_actions(robot_actions)

                # Compute action loss
                # (B, horizon, action_dim, vocab_size) vs (B, horizon, action_dim)
                action_logits_flat = action_logits.view(-1, model.action_vocab_size)
                action_tokens_flat = action_tokens.view(-1)

                loss = action_loss_fn(action_logits_flat, action_tokens_flat)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model
```

**Training Performance**:
- **RT-2 (PaLI-X 55B)**: ~1 week on 512 TPU v4 chips
- **Data mixture**: 50% robot data (Open X-Embodiment), 50% web data
- **Batch size**: 1024-2048 (gradient accumulation)
- **Learning rate**: 1e-5 to 1e-4 (warm up 10%)

---

## 4. Embodied AI Applications

### 4.1 Robot Manipulation with VLAs

```python
import numpy as np
from typing import Dict, Any

class EmbodiedAgent:
    """
    Closed-loop robot control using VLA model
    """
    def __init__(
        self,
        vla_model: VisionLanguageActionModel,
        camera_interface,
        robot_interface,
        tokenizer,
        control_freq: int = 10  # Hz
    ):
        self.model = vla_model.eval()
        self.camera = camera_interface
        self.robot = robot_interface
        self.tokenizer = tokenizer
        self.control_freq = control_freq

    def execute_task(
        self,
        instruction: str,
        max_steps: int = 100,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Execute high-level instruction in closed loop

        Example: "Pick up the apple and place it in the bowl"
        """
        trajectory = []

        for step in range(max_steps):
            # Get current observation
            image = self.camera.get_rgb_image()  # (H, W, 3)
            image_tensor = self.preprocess_image(image).to(device)

            # Predict action
            with torch.no_grad():
                action = self.model.predict_action(
                    pixel_values=image_tensor.unsqueeze(0),
                    text_prompt=instruction,
                    tokenizer=self.tokenizer,
                    device=device
                )

            # Convert to robot commands
            action_np = action.cpu().numpy()[0]  # (7,)

            # Execute on robot (delta actions in range [-1, 1])
            # Assuming: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            self.robot.execute_delta_action(
                delta_pos=action_np[:3] * 0.01,  # Scale to meters
                delta_rot=action_np[3:6] * 0.1,  # Scale to radians
                gripper=action_np[6]  # -1 (close) to 1 (open)
            )

            # Record trajectory
            trajectory.append({
                "step": step,
                "image": image,
                "action": action_np,
                "robot_state": self.robot.get_state()
            })

            # Check termination (e.g., gripper closed and object grasped)
            if self.check_termination():
                break

            # Control loop timing
            self.sleep_until_next_tick()

        return {
            "trajectory": trajectory,
            "success": self.check_success(),
            "num_steps": len(trajectory)
        }

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert camera image to model input
        """
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transform(image)
```

### 4.2 Real-World Deployment: Physical Intelligence π0

From Physical Intelligence's π0 paper (2025):

**Key Innovation**: Flow matching for continuous action generation (vs discrete tokens)

```python
class Pi0FlowModel(nn.Module):
    """
    π0 architecture: VLM + flow matching for actions

    Based on: https://www.physicalintelligence.company/download/pi0.pdf
    """
    def __init__(
        self,
        vlm_backbone: str = "meta-llama/Llama-3.2-90B-Vision",
        action_dim: int = 7,
        flow_steps: int = 50
    ):
        super().__init__()

        from transformers import AutoModel

        # Pre-trained VLM
        self.vlm = AutoModel.from_pretrained(vlm_backbone)
        hidden_dim = self.vlm.config.hidden_size

        # Flow matching network (learns vector field)
        self.flow_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + 1, 512),  # +1 for time
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.action_dim = action_dim
        self.flow_steps = flow_steps

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        t: torch.Tensor,  # Flow time (0 to 1)
        x_t: torch.Tensor  # Current flow state
    ):
        """
        Predict velocity field at time t
        """
        # VLM conditioning
        vlm_out = self.vlm(pixel_values=image, input_ids=text)
        context = vlm_out.last_hidden_state[:, 0, :]  # (B, hidden_dim)

        # Flow network input: [context, x_t, t]
        flow_input = torch.cat([context, x_t, t.unsqueeze(-1)], dim=-1)

        # Predict velocity
        v_t = self.flow_net(flow_input)

        return v_t

    @torch.no_grad()
    def sample_action(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        device: str = "cuda"
    ):
        """
        Generate action via flow matching (Euler integration)
        """
        batch_size = image.shape[0]

        # Start from noise
        x = torch.randn(batch_size, self.action_dim).to(device)

        # Integrate flow
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t = torch.full((batch_size,), step * dt).to(device)

            # Predict velocity
            v = self.forward(image, text, t, x)

            # Euler step
            x = x + v * dt

        return x  # Final action
```

**π0 Performance** (from paper):
- **Generalist policy**: Single model for laundry folding, table cleaning, assembly
- **Success rate**: 67% on novel objects (vs 31% for RT-2)
- **Deployment**: Real robots in diverse environments

---

## 5. 🚂 TRAIN STATION: Action = Affordance = Embodied = Participatory

### 5.1 The Unification

**Action-conditioned VLMs unify multiple perspectives**:

| Concept | Mapping | Evidence |
|---------|---------|----------|
| **Gibson Affordances** | Action = affordance detection | VLA predicts "what can I do with this object?" |
| **Active Inference** | Action = free energy minimization | VLA selects actions that reduce prediction error |
| **Embodied Cognition** | Action = perceptual grounding | VLA learns actions through physical interaction |
| **Participatory Sense-Making** | Action = world co-creation | VLA's actions change the environment it perceives |

```python
# Pseudocode showing the unification
class UnifiedVLA:
    """
    VLA as affordance detector + active inference agent + embodied learner
    """
    def predict_action(self, image, instruction):
        # 1. Affordance detection (Gibson)
        affordances = self.detect_affordances(image)  # "Can grasp this"

        # 2. Expected free energy (Friston)
        actions = self.generate_action_candidates()
        efe = [self.compute_efe(a, affordances) for a in actions]

        # 3. Select action that minimizes EFE
        best_action = actions[np.argmin(efe)]

        # 4. Execute (embodied)
        observation_prime = self.execute(best_action)

        # 5. Update world model (participatory)
        self.update_model(observation_prime)

        return best_action
```

### 5.2 Action Spaces as Embedding Spaces

**Key insight**: VLA action spaces are semantic embeddings!

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_action_space(vla_model, scenarios, device="cuda"):
    """
    Show that action space has semantic structure
    """
    actions = []
    labels = []

    for scenario in scenarios:
        image, text = scenario["image"], scenario["text"]

        with torch.no_grad():
            action = vla_model.predict_action(
                pixel_values=image.to(device),
                text_prompt=text,
                tokenizer=tokenizer,
                device=device
            )

        actions.append(action.cpu().numpy())
        labels.append(text)

    # PCA to 2D
    actions_np = np.vstack(actions)
    pca = PCA(n_components=2)
    actions_2d = pca.fit_transform(actions_np)

    # Plot
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(actions_2d[i, 0], actions_2d[i, 1])
        plt.text(actions_2d[i, 0], actions_2d[i, 1], label, fontsize=8)

    plt.xlabel("Action PC 1")
    plt.ylabel("Action PC 2")
    plt.title("VLA Action Space Embedding (semantic structure)")
    plt.show()

# Example scenarios
scenarios = [
    {"text": "pick up the apple", "image": apple_img},
    {"text": "pick up the banana", "image": banana_img},
    {"text": "place in bowl", "image": bowl_img},
    {"text": "open the drawer", "image": drawer_img},
    # Actions for similar objects cluster together!
]
```

**Observation**: Actions for semantically similar instructions cluster in action space, showing VLA learned meaningful action embeddings.

---

## 6. ARR-COC-0-1: Action-Relevant Token Allocation

### 6.1 Connecting to Relevance Realization

**Question**: How does ARR-COC allocate attention to *action-relevant* tokens?

```python
class ActionRelevanceAttention(nn.Module):
    """
    Attention mechanism that weights tokens by action relevance

    Connects to ARR-COC's relevance realization:
    - Relevance = what matters for the action
    - Attention weights = relevance scores
    """
    def __init__(self, hidden_dim: int):
        super().__init__()

        # Query: current action intent
        self.action_query = nn.Linear(hidden_dim, hidden_dim)

        # Key/Value: visual/language tokens
        self.token_key = nn.Linear(hidden_dim, hidden_dim)
        self.token_value = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        action_intent: torch.Tensor,  # (B, hidden_dim) - goal encoding
        token_embeds: torch.Tensor    # (B, N, hidden_dim) - image/text tokens
    ):
        """
        Compute action-relevant attention
        """
        # Query from action intent
        Q = self.action_query(action_intent).unsqueeze(1)  # (B, 1, D)

        # Keys and values from tokens
        K = self.token_key(token_embeds)    # (B, N, D)
        V = self.token_value(token_embeds)  # (B, N, D)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.shape[-1])
        relevance_weights = torch.softmax(scores, dim=-1)  # (B, 1, N)

        # Aggregate relevant tokens
        action_relevant = torch.matmul(relevance_weights, V)  # (B, 1, D)

        return action_relevant.squeeze(1), relevance_weights.squeeze(1)


# Example: Action-conditioned dialogue system
class DialogueWithActionContext:
    """
    ARR-COC-style dialogue that considers *what you're doing*
    """
    def __init__(self, vla_model, dialogue_model):
        self.vla = vla_model
        self.dialogue = dialogue_model
        self.action_relevance = ActionRelevanceAttention(hidden_dim=512)

    def generate_response(
        self,
        user_utterance: str,
        current_image: torch.Tensor,
        current_action: str  # "folding laundry", "cooking", etc.
    ):
        """
        Dialogue response considers current embodied action
        """
        # Encode action intent
        action_intent = self.vla.encode_action_intent(current_action)

        # Encode user utterance + image
        tokens = self.dialogue.encode(user_utterance, current_image)

        # Attention weighted by action relevance
        relevant_tokens, weights = self.action_relevance(action_intent, tokens)

        # Generate response using action-relevant context
        response = self.dialogue.generate(relevant_tokens)

        return response, weights

# Usage
dialogue = DialogueWithActionContext(vla_model, dialogue_model)

# User asks: "What am I holding?"
# Action context: "folding laundry"
# VLA attends to hands + fabric in image (action-relevant!)
response, attn = dialogue.generate_response(
    user_utterance="What am I holding?",
    current_image=camera_feed,
    current_action="folding laundry"
)
# Response: "You're holding a white t-shirt, about to fold it."
```

### 6.2 Relevance = Affordance in ARR-COC

**Key connection**: In embodied dialogue, relevance *is* affordance!

- **Relevance Realization** = "What matters for my current goals?"
- **Affordance Detection** = "What actions can I take?"
- **Unified in VLA**: Relevant tokens = tokens that afford successful actions

```
Traditional VLM:     Image → Text (static, passive)
Action-VLM (VLA):   Image → Action (dynamic, embodied)
ARR-COC + VLA:      Image + Dialogue → Action-Relevant Response

Relevance Criterion: Does this token help me act successfully?
```

**Implementation insight**: ARR-COC could use VLA's action prediction loss as a relevance signal!

---

## 7. Implementation Notes & Performance

### 7.1 Model Sizes and Latency

| Model | Params | Inference (GPU) | Training Time |
|-------|--------|-----------------|---------------|
| RT-2 (PaLM-E 12B) | 12B | ~100ms | 3 days (128 GPUs) |
| RT-2 (PaLI-X 55B) | 55B | ~500ms | 7 days (512 TPUs) |
| π0 (Llama 90B) | 90B | ~800ms | 14 days (1024 GPUs) |
| OpenVLA (7B) | 7B | ~50ms | 1 day (64 GPUs) |

**Optimization strategies**:
- **Quantization**: INT8 reduces latency by 2-3x (50ms → 20ms for 7B model)
- **KV caching**: Reuse past keys/values for autoregressive generation
- **Batching**: Process multiple robot queries in parallel

### 7.2 Action Representation Trade-offs

| Method | Pros | Cons |
|--------|------|------|
| **Discrete tokens** (RT-2) | Simple, works with LM head | Quantization error, high-dim curse |
| **Continuous (diffusion)** (π0) | Smooth, high-fidelity | Slower inference (50 steps) |
| **Hybrid** (OpenVLA) | Best of both | More complex training |

### 7.3 Data Requirements

**Open X-Embodiment Dataset** (used by RT-2):
- **Size**: 1M+ robot trajectories across 22 embodiments
- **Diversity**: Manipulation, locomotion, mobile manipulation
- **Format**: (observation, language, action, reward)

**Co-training data mixture** (recommended):
- 50% robot data (trajectories)
- 30% web vision-language (VQA, captioning)
- 20% video data (temporal grounding)

---

## Sources

**Source Documents:**
- Dialogue 67 concepts (lines 1-500): Embodied AI, affordance detection, participatory sense-making

**Web Research:**

**RT-2 (Google DeepMind, 2023)**:
- [RT-2 Project Page](https://robotics-transformer2.github.io/) - Official demos and architecture
- [RT-2 Paper (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818) - Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
- [Google Blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) - RT-2 announcement (accessed 2025-11-23)

**Vision-Language-Action Surveys**:
- [VLA Survey (arXiv:2405.14093)](https://arxiv.org/abs/2405.14093) - Ma et al., "A Survey on Vision-Language-Action Models for Embodied AI" (2024)
- [VLA-Robotics Survey](https://vla-survey.github.io/) - Comprehensive VLA review site (accessed 2025-11-23)

**Physical Intelligence π0**:
- [π0 Paper](https://www.physicalintelligence.company/download/pi0.pdf) - "π0: A Vision-Language-Action Flow Model for Generalist Robot Control" (2025)

**Open Source Implementations**:
- [OpenVLA GitHub](https://github.com/openvla/openvla) - Open source VLA training code
- [RT-2 Implementation](https://github.com/kyegomez/RT-2) - Community PyTorch implementation

**Additional References**:
- Open X-Embodiment dataset - Multi-robot trajectory data
- LearnOpenCV VLA Tutorial - Practical VLA implementation guide
- Stanford CS231n Lecture 17 - Robot Learning (2025)

**Related ARR-COC Concepts**:
- Affordance detection → ml-affordances/00-affordance-detection.md
- World models → ml-affordances/04-world-models-affordances.md
- Active inference → ml-active-inference/00-active-inference-pytorch.md

---

**Created**: 2025-11-23
**PART 32 of 42** - ML-HEAVY Knowledge Expansion (ZEUS Pattern)
