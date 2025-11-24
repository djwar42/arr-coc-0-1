# Trust Calibration Methods for AI Systems

## Overview

Trust calibration is the process of aligning human trust in AI systems with the actual trustworthiness and reliability of those systems. When trust is well-calibrated, humans appropriately rely on AI assistance when it's competent and appropriately distrust it when it fails. Poor calibration leads to either **overtrust** (accepting flawed outputs) or **undertrust** (rejecting valuable insights).

Trust calibration is crucial for human-AI collaboration in high-stakes domains like medical diagnosis, autonomous vehicles, and financial decision-making. Unlike simple accuracy metrics, trust calibration measures whether AI systems provide **honest signals** about their own reliability.

**Key Challenge**: AI systems (especially deep neural networks) are often **overconfident** - they output high probability scores even when wrong. This miscalibration undermines human ability to assess when to trust AI recommendations.

---

## 1. Feature-Specific Trust

### Concept

Feature-specific trust recognizes that AI systems may be trustworthy for some capabilities while unreliable for others. Rather than treating trust as monolithic, feature-specific approaches allow **granular trust assessment per capability**.

**Example**: A medical AI might be highly trustworthy for detecting lung nodules but unreliable for diagnosing rare diseases. Users need separate trust scores for each capability.

### Physical AI Systems

From [Feature-Specific Trust Calibration in Physical AI Systems (ICIS 2025)](https://aisel.aisnet.org/icis2025/user_behav/user_behav/14/):

Physical AI systems (robots, autonomous vehicles, drones) require feature-specific trust because:
- **Temporal dynamics**: Trust must update in real-time as conditions change
- **Capability heterogeneity**: Different features have different reliability profiles
- **Context-dependent performance**: Same feature may perform differently in different environments

**Implementation approach**:
```python
class FeatureSpecificTrustTracker:
    def __init__(self, features):
        self.trust_scores = {f: 0.5 for f in features}  # Initialize at neutral
        self.performance_history = {f: [] for f in features}

    def update_trust(self, feature, success, context):
        """Update trust for specific feature based on outcome"""
        # Exponential moving average with context weighting
        alpha = 0.1  # Learning rate
        context_weight = self.compute_context_similarity(feature, context)

        outcome = 1.0 if success else 0.0
        self.trust_scores[feature] = (
            (1 - alpha) * self.trust_scores[feature] +
            alpha * outcome * context_weight
        )

        self.performance_history[feature].append({
            'outcome': outcome,
            'context': context,
            'timestamp': time.time()
        })

    def get_trust(self, feature, current_context):
        """Get trust score for feature in current context"""
        base_trust = self.trust_scores[feature]
        context_adjustment = self.compute_context_adjustment(
            feature, current_context
        )
        return base_trust * context_adjustment
```

**Key insight**: Trust should be **feature-specific** and **context-aware**, not a single global score.

---

## 2. Dynamic Trust Adjustment

### Adaptive Trust Calibration

From [Dynamic Trust Calibration Using Contextual Bandits (arXiv:2509.23497, 2025)](https://arxiv.org/abs/2509.23497):

Dynamic trust calibration uses **contextual bandits** to learn when to trust AI contributions based on context:

**Contextual Bandit Formulation**:
- **Actions**: {Trust AI, Distrust AI}
- **Context**: Current problem features, AI confidence, past performance
- **Reward**: Decision quality (correctness, value gained)

```python
import torch
import torch.nn as nn

class DynamicTrustIndicator(nn.Module):
    """Learns when to trust AI using contextual bandits"""

    def __init__(self, context_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Trust vs Distrust
            nn.Softmax(dim=-1)
        )

    def forward(self, context):
        """Output trust recommendation probability"""
        return self.network(context)

    def should_trust(self, context, ai_confidence, threshold=0.5):
        """Decide whether to trust AI in current context"""
        full_context = torch.cat([context, ai_confidence.unsqueeze(-1)], dim=-1)
        probs = self.forward(full_context)
        return probs[:, 0] > threshold  # Trust if P(trust) > threshold


class TrustCalibrationTrainer:
    """Train dynamic trust indicator with rewards"""

    def __init__(self, indicator, lr=0.001):
        self.indicator = indicator
        self.optimizer = torch.optim.Adam(indicator.parameters(), lr=lr)

    def update(self, context, action_taken, reward):
        """Update based on decision outcome"""
        self.optimizer.zero_grad()

        # Get action probabilities
        probs = self.indicator(context)

        # Compute policy gradient loss
        action_idx = 0 if action_taken == 'trust' else 1
        log_prob = torch.log(probs[action_idx])
        loss = -log_prob * reward  # Maximize reward

        loss.backward()
        self.optimizer.step()
```

**Performance gains**: Study showed 10-38% improvement in decision quality when using dynamic trust calibration across three datasets (disease diagnosis, recidivism prediction, financial lending).

### Real-Time Trust Scoring

From [Dynamic Trust Scoring in IAM (Identity Management Institute, 2025)](https://identitymanagementinstitute.org/dynamic-trust-scoring-in-iam/):

Machine learning models continuously learn baseline behavior and flag deviations:

```python
class RealTimeTrustScorer:
    """Continuous trust score updates based on behavior monitoring"""

    def __init__(self, baseline_model):
        self.baseline = baseline_model  # Pre-trained on normal behavior
        self.alert_threshold = 0.3  # Trust score threshold for alerts

    def compute_trust_score(self, current_behavior, resource_sensitivity):
        """Compute trust score based on deviation from baseline"""

        # Measure behavioral deviation
        baseline_features = self.baseline.get_features()
        deviation = self.compute_deviation(current_behavior, baseline_features)

        # Adjust threshold based on resource sensitivity
        dynamic_threshold = self.alert_threshold * resource_sensitivity

        # Convert deviation to trust score (0-1)
        trust_score = max(0, 1.0 - deviation / dynamic_threshold)

        return trust_score, deviation > dynamic_threshold  # score, alert

    def compute_deviation(self, behavior, baseline):
        """Statistical deviation from normal behavior patterns"""
        # Mahalanobis distance or other anomaly score
        return torch.norm(behavior - baseline.mean) / (baseline.std + 1e-8)
```

**Key features**:
- **Continuous monitoring**: Trust updates in real-time, not static
- **Adaptive thresholds**: Adjust based on resource criticality
- **Behavioral baselines**: Learn what "normal" looks like per user/system

---

## 3. Uncertainty Quantification Techniques

### Why Uncertainty Matters for Trust

Well-calibrated uncertainty enables informed trust decisions. When an AI says "90% confident," it should be correct 90% of the time. Most neural networks are **overconfident** - they output 90%+ probabilities even when making errors.

### Calibration Metrics

#### Expected Calibration Error (ECE)

From [On Calibration of Modern Neural Networks (ICML 2017)](https://dl.acm.org/doi/10.5555/3305381.3305518):

ECE measures the difference between predicted confidence and actual accuracy:

```python
import torch

def expected_calibration_error(confidences, predictions, labels, n_bins=15):
    """
    Compute Expected Calibration Error

    Args:
        confidences: Model confidence scores [N]
        predictions: Predicted classes [N]
        labels: True labels [N]
        n_bins: Number of bins for calibration curve

    Returns:
        ECE score (lower is better, 0 is perfect calibration)
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = (predictions == labels).float()

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            # Weight by proportion of samples
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def compute_calibration_curve(confidences, predictions, labels, n_bins=15):
    """Compute calibration curve data for plotting"""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    accuracies_per_bin = []
    confidences_per_bin = []
    counts_per_bin = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            accuracies_per_bin.append((predictions[in_bin] == labels[in_bin]).float().mean())
            confidences_per_bin.append(confidences[in_bin].mean())
            counts_per_bin.append(in_bin.sum().item())
        else:
            accuracies_per_bin.append(0)
            confidences_per_bin.append(0)
            counts_per_bin.append(0)

    return {
        'bin_centers': bin_centers,
        'accuracies': torch.tensor(accuracies_per_bin),
        'confidences': torch.tensor(confidences_per_bin),
        'counts': counts_per_bin
    }
```

**Perfect calibration**: ECE = 0 (predicted confidence exactly matches actual accuracy)

#### Brier Score

The Brier score measures both calibration and sharpness:

```python
def brier_score(probabilities, labels):
    """
    Compute Brier score (proper scoring rule)

    Args:
        probabilities: Predicted probabilities [N, num_classes]
        labels: True labels [N]

    Returns:
        Brier score (lower is better, 0 is perfect)
    """
    N, num_classes = probabilities.shape

    # Convert labels to one-hot
    one_hot = torch.zeros_like(probabilities)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)

    # Mean squared error between predicted probs and true probs
    brier = torch.mean((probabilities - one_hot) ** 2)

    return brier.item()


def brier_score_decomposition(probabilities, labels):
    """
    Decompose Brier score into reliability, resolution, uncertainty

    Brier = Reliability - Resolution + Uncertainty
    """
    N = len(labels)
    num_classes = probabilities.shape[1]

    # Get predicted classes and confidences
    confidences, predictions = probabilities.max(dim=1)

    # Overall accuracy
    base_rate = labels.float().mean()

    # Uncertainty: inherent difficulty of problem
    uncertainty = base_rate * (1 - base_rate)

    # Reliability: how far predicted probs are from observed frequencies
    # (computed via binning like ECE)
    n_bins = 10
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        bin_mask = (confidences > i/n_bins) & (confidences <= (i+1)/n_bins)
        if bin_mask.sum() > 0:
            bin_accuracy = (predictions[bin_mask] == labels[bin_mask]).float().mean()
            bin_confidence = confidences[bin_mask].mean()
            bin_proportion = bin_mask.float().mean()

            reliability += bin_proportion * (bin_confidence - bin_accuracy) ** 2
            resolution += bin_proportion * (bin_accuracy - base_rate) ** 2

    return {
        'brier': reliability - resolution + uncertainty,
        'reliability': reliability.item(),
        'resolution': resolution.item(),
        'uncertainty': uncertainty.item()
    }
```

**Key difference**: ECE focuses on calibration, Brier score combines calibration + discrimination.

### Calibration Methods

#### Temperature Scaling

From [Calibrating Language Models with Adaptive Temperature Scaling (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1007.pdf):

Temperature scaling is a **single-parameter post-hoc calibration** method that rescales logits:

```python
class TemperatureScaler(nn.Module):
    """Post-hoc calibration via temperature scaling"""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        """Apply temperature scaling to logits"""
        return logits / self.temperature

    def calibrate(self, val_logits, val_labels, lr=0.01, max_iters=50):
        """Learn optimal temperature on validation set"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nn.functional.cross_entropy(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        print(f"Optimal temperature: {self.temperature.item():.3f}")
        return self.temperature.item()


# Usage example
def apply_temperature_scaling(model, val_loader):
    """Calibrate model on validation set"""

    # Collect validation logits
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs)
            all_logits.append(logits)
            all_labels.append(labels)

    val_logits = torch.cat(all_logits)
    val_labels = torch.cat(all_labels)

    # Fit temperature
    temp_scaler = TemperatureScaler()
    optimal_temp = temp_scaler.calibrate(val_logits, val_labels)

    # Wrap model with temperature scaling
    class CalibratedModel(nn.Module):
        def __init__(self, base_model, temperature):
            super().__init__()
            self.base_model = base_model
            self.temperature = temperature

        def forward(self, x):
            logits = self.base_model(x)
            return logits / self.temperature

    return CalibratedModel(model, optimal_temp)
```

**Why it works**:
- **Overconfident models** have T < 1 (sharpen probabilities less)
- **Underconfident models** have T > 1 (sharpen probabilities more)
- **Single parameter** prevents overfitting on validation set

#### Adaptive Temperature Scaling

From [Calibrating Language Models with Adaptive Temperature Scaling (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1007.pdf):

Adaptive temperature scaling learns **input-dependent temperatures**:

```python
class AdaptiveTemperatureScaler(nn.Module):
    """Learn temperature as function of input features"""

    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.temp_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive temperature
        )

    def forward(self, features, logits):
        """Compute adaptive temperature and scale logits"""
        temperature = self.temp_network(features) + 0.5  # Min temp = 0.5
        scaled_logits = logits / temperature
        return scaled_logits, temperature


# Training loop
def train_adaptive_temperature(model, temp_scaler, train_loader, val_loader, epochs=10):
    """Train adaptive temperature scaler"""

    optimizer = torch.optim.Adam(temp_scaler.parameters(), lr=0.001)

    for epoch in range(epochs):
        temp_scaler.train()

        for inputs, labels in train_loader:
            # Get features and logits from base model
            features = model.get_features(inputs)  # Intermediate representation
            logits = model.get_logits(features)

            # Apply adaptive temperature
            scaled_logits, temps = temp_scaler(features, logits)

            # Optimize for both accuracy and calibration
            ce_loss = nn.functional.cross_entropy(scaled_logits, labels)

            # Add regularization to prevent extreme temperatures
            temp_reg = 0.01 * ((temps - 1.0) ** 2).mean()

            loss = ce_loss + temp_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate calibration on validation set
        val_ece = evaluate_calibration(model, temp_scaler, val_loader)
        print(f"Epoch {epoch}: Val ECE = {val_ece:.4f}")
```

**Improvement**: 31% better calibration than fixed temperature scaling on LLMs (EMNLP 2024).

#### Platt Scaling

Platt scaling learns a logistic regression on top of model outputs:

```python
from sklearn.linear_model import LogisticRegression

def platt_scaling(val_scores, val_labels):
    """
    Fit Platt scaling (logistic calibration)

    Args:
        val_scores: Model scores/logits [N, num_classes]
        val_labels: True labels [N]

    Returns:
        Fitted calibrator
    """
    calibrator = LogisticRegression()
    calibrator.fit(val_scores, val_labels)

    return calibrator


class PlattScaledModel(nn.Module):
    """Model with Platt scaling calibration"""

    def __init__(self, base_model, calibrator):
        super().__init__()
        self.base_model = base_model
        self.calibrator = calibrator

    def forward(self, x):
        with torch.no_grad():
            logits = self.base_model(x)

        # Apply Platt scaling
        calibrated_probs = self.calibrator.predict_proba(logits.cpu().numpy())

        return torch.tensor(calibrated_probs)
```

**Comparison**:
- **Temperature scaling**: 1 parameter, better for small validation sets
- **Platt scaling**: 2 parameters per class, more flexible but risks overfitting

#### Ensemble Calibration

From [Achieving Well-Informed Decision-Making in Drug Discovery (NeurIPS 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11881400/):

Ensemble methods improve calibration by averaging diverse models:

```python
class EnsembleCalibrator:
    """Calibrate via model ensemble averaging"""

    def __init__(self, models):
        self.models = models

    def predict_calibrated(self, x):
        """Average predictions from ensemble"""
        all_probs = []

        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs)

        # Average probabilities
        avg_probs = torch.stack(all_probs).mean(dim=0)

        return avg_probs

    def predict_with_uncertainty(self, x):
        """Return mean prediction + uncertainty estimate"""
        all_probs = []

        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs)

        # Predictive mean
        mean_probs = all_probs.mean(dim=0)

        # Predictive uncertainty (variance)
        uncertainty = all_probs.var(dim=0)

        return mean_probs, uncertainty
```

**Why ensembles help calibration**: Diverse errors average out, uncertainty estimates are more reliable.

---

## 4. Monitoring Methods for Trust Violations

### Real-Time Monitoring Systems

From [Real-Time AI Governance: Monitoring, Compliance (Relyance AI 2025)](https://www.relyance.ai/blog/ai-governance):

Continuous monitoring detects when AI systems deviate from expected behavior:

```python
class AITrustMonitor:
    """Monitor AI system for trust violations"""

    def __init__(self, model, baseline_metrics):
        self.model = model
        self.baseline = baseline_metrics
        self.alert_log = []

    def monitor_predictions(self, inputs, outputs, ground_truth=None):
        """Check predictions for trust violations"""

        violations = []

        # Check 1: Confidence calibration
        confidences, predictions = outputs.max(dim=-1)
        if ground_truth is not None:
            accuracy = (predictions == ground_truth).float().mean()
            avg_confidence = confidences.mean()

            if abs(avg_confidence - accuracy) > 0.15:  # Miscalibration threshold
                violations.append({
                    'type': 'calibration_drift',
                    'severity': 'high',
                    'confidence': avg_confidence.item(),
                    'accuracy': accuracy.item()
                })

        # Check 2: Distribution shift
        input_stats = self.compute_input_statistics(inputs)
        if self.detect_distribution_shift(input_stats):
            violations.append({
                'type': 'distribution_shift',
                'severity': 'medium',
                'details': input_stats
            })

        # Check 3: Output uncertainty spike
        output_entropy = self.compute_entropy(outputs)
        if output_entropy > self.baseline['max_entropy'] * 1.5:
            violations.append({
                'type': 'high_uncertainty',
                'severity': 'low',
                'entropy': output_entropy.item()
            })

        # Log violations
        if violations:
            self.alert_log.append({
                'timestamp': time.time(),
                'violations': violations
            })

        return violations

    def compute_entropy(self, probs):
        """Compute predictive entropy"""
        return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

    def detect_distribution_shift(self, current_stats):
        """Detect if input distribution has shifted"""
        baseline_mean = self.baseline['input_mean']
        baseline_std = self.baseline['input_std']

        # Compute KL divergence or Wasserstein distance
        # Simplified: compare statistics
        mean_shift = torch.norm(current_stats['mean'] - baseline_mean)
        std_shift = torch.norm(current_stats['std'] - baseline_std)

        return mean_shift > 0.5 or std_shift > 0.3
```

### Anomaly Detection for Trust

```python
class TrustAnomalyDetector:
    """Detect anomalous behavior patterns indicating trust violations"""

    def __init__(self, normal_behavior_data):
        # Fit autoencoder on normal behavior
        self.autoencoder = self.train_autoencoder(normal_behavior_data)
        self.threshold = self.compute_threshold(normal_behavior_data)

    def train_autoencoder(self, data, hidden_dim=64, latent_dim=16):
        """Train autoencoder to learn normal behavior"""
        input_dim = data.shape[1]

        encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        autoencoder = nn.Sequential(encoder, decoder)

        # Train to reconstruct normal behavior
        optimizer = torch.optim.Adam(autoencoder.parameters())

        for epoch in range(50):
            reconstructed = autoencoder(data)
            loss = nn.functional.mse_loss(reconstructed, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return autoencoder

    def compute_threshold(self, data, percentile=95):
        """Set anomaly threshold based on normal data"""
        with torch.no_grad():
            reconstructed = self.autoencoder(data)
            reconstruction_errors = ((data - reconstructed) ** 2).sum(dim=1)
            threshold = torch.quantile(reconstruction_errors, percentile / 100)

        return threshold

    def is_anomalous(self, behavior):
        """Check if behavior is anomalous (potential trust violation)"""
        with torch.no_grad():
            reconstructed = self.autoencoder(behavior)
            error = ((behavior - reconstructed) ** 2).sum(dim=1)

        return error > self.threshold, error.item()
```

---

## 5. Implementation Examples

### Complete Trust-Calibrated Model

```python
class TrustCalibratedModel(nn.Module):
    """Full model with trust calibration and monitoring"""

    def __init__(self, base_model, calibration_method='temperature'):
        super().__init__()
        self.base_model = base_model
        self.calibration_method = calibration_method

        # Calibration components
        if calibration_method == 'temperature':
            self.calibrator = TemperatureScaler()
        elif calibration_method == 'adaptive':
            feature_dim = base_model.feature_dim
            self.calibrator = AdaptiveTemperatureScaler(feature_dim)

        # Trust tracking
        self.trust_tracker = FeatureSpecificTrustTracker(
            features=['prediction', 'uncertainty', 'explanation']
        )

        # Monitoring
        self.monitor = AITrustMonitor(base_model, baseline_metrics={})

    def forward(self, x, return_trust_info=False):
        """Forward pass with calibration and trust estimation"""

        # Base model prediction
        logits = self.base_model(x)

        # Apply calibration
        if self.calibration_method == 'temperature':
            calibrated_logits = self.calibrator(logits)
        elif self.calibration_method == 'adaptive':
            features = self.base_model.get_features(x)
            calibrated_logits, temps = self.calibrator(features, logits)

        # Get calibrated probabilities
        probs = torch.softmax(calibrated_logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)

        if return_trust_info:
            # Compute trust metrics
            uncertainty = self.compute_uncertainty(probs)
            trust_score = self.estimate_trust(confidences, uncertainty)

            return {
                'predictions': predictions,
                'probabilities': probs,
                'confidences': confidences,
                'uncertainty': uncertainty,
                'trust_score': trust_score
            }

        return probs

    def compute_uncertainty(self, probs):
        """Compute predictive uncertainty via entropy"""
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(probs.shape[-1]))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def estimate_trust(self, confidences, uncertainty):
        """Estimate trust score from confidence and uncertainty"""
        # High confidence + low uncertainty = high trust
        # Low confidence or high uncertainty = low trust
        trust = confidences * (1 - uncertainty)

        return trust

    def calibrate_on_validation(self, val_loader):
        """Calibrate model on validation set"""
        all_logits = []
        all_labels = []

        self.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                logits = self.base_model(inputs)
                all_logits.append(logits)
                all_labels.append(labels)

        val_logits = torch.cat(all_logits)
        val_labels = torch.cat(all_labels)

        # Calibrate
        if self.calibration_method == 'temperature':
            self.calibrator.calibrate(val_logits, val_labels)
        elif self.calibration_method == 'adaptive':
            # Adaptive requires features - more complex training
            pass

        # Update baseline metrics for monitoring
        with torch.no_grad():
            calibrated_logits = self.calibrator(val_logits)
            probs = torch.softmax(calibrated_logits, dim=-1)

            self.monitor.baseline = {
                'input_mean': val_logits.mean(dim=0),
                'input_std': val_logits.std(dim=0),
                'max_entropy': -(probs * torch.log(probs + 1e-10)).sum(dim=-1).max()
            }


# Usage example
def train_with_trust_calibration(model, train_loader, val_loader, test_loader):
    """Complete training pipeline with trust calibration"""

    # Train base model
    print("Training base model...")
    train_base_model(model.base_model, train_loader, epochs=10)

    # Calibrate on validation set
    print("Calibrating model...")
    model.calibrate_on_validation(val_loader)

    # Evaluate calibration
    print("Evaluating calibration on test set...")
    all_probs = []
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs, return_trust_info=True)
            all_probs.append(output['probabilities'])
            all_preds.append(output['predictions'])
            all_labels.append(labels)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # Compute metrics
    confidences = probs.max(dim=-1)[0]
    ece = expected_calibration_error(confidences, preds, labels)
    brier = brier_score(probs, labels)
    accuracy = (preds == labels).float().mean()

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ECE: {ece:.4f}")
    print(f"Test Brier Score: {brier:.4f}")

    return model
```

---

## 6. ARR-COC Application: Trust Calibration for Relevance Realization

### Calibrating Relevance Confidence

In ARR-COC, trust calibration ensures that relevance scores honestly reflect actual relevance:

```python
class CalibratedRelevanceRealizer:
    """ARR-COC with trust-calibrated relevance scores"""

    def __init__(self, base_realizer):
        self.base_realizer = base_realizer
        self.temperature = nn.Parameter(torch.ones(1))

    def realize_relevance_calibrated(self, visual_features, query_embed):
        """Compute calibrated relevance scores"""

        # Base relevance scores (propositional + perspectival + participatory)
        P_prop = self.base_realizer.propositional_scorer(visual_features)
        P_pers = self.base_realizer.perspectival_scorer(visual_features)
        P_part = self.base_realizer.participatory_scorer(visual_features, query_embed)

        # Combined relevance (uncalibrated)
        relevance_logits = torch.stack([P_prop, P_pers, P_part], dim=-1)
        relevance_scores = torch.sigmoid(relevance_logits)

        # Apply temperature scaling for calibration
        calibrated_logits = relevance_logits / self.temperature
        calibrated_scores = torch.sigmoid(calibrated_logits)

        # Compute trust score based on confidence and uncertainty
        confidence = calibrated_scores.max(dim=-1)[0]
        uncertainty = self.compute_relevance_uncertainty(calibrated_scores)
        trust_score = confidence * (1 - uncertainty)

        return {
            'relevance_scores': calibrated_scores,
            'trust_score': trust_score,
            'confidence': confidence,
            'uncertainty': uncertainty
        }

    def compute_relevance_uncertainty(self, scores):
        """Estimate uncertainty in relevance assessment"""
        # Variance across three ways of knowing
        variance = scores.var(dim=-1)

        # Normalize to [0, 1]
        max_variance = 0.25  # Max variance when all scores = 0.5
        normalized_uncertainty = torch.sqrt(variance / max_variance)

        return normalized_uncertainty

    def calibrate_relevance(self, val_features, val_queries, val_relevance_labels):
        """Calibrate relevance scores on validation set"""

        # Collect uncalibrated scores
        all_logits = []

        with torch.no_grad():
            for feats, query in zip(val_features, val_queries):
                query_embed = self.base_realizer.query_encoder(query)

                P_prop = self.base_realizer.propositional_scorer(feats)
                P_pers = self.base_realizer.perspectival_scorer(feats)
                P_part = self.base_realizer.participatory_scorer(feats, query_embed)

                logits = torch.stack([P_prop, P_pers, P_part], dim=-1).mean(dim=-1)
                all_logits.append(logits)

        all_logits = torch.cat(all_logits)

        # Optimize temperature to minimize calibration error
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)

        def eval_loss():
            optimizer.zero_grad()
            calibrated_probs = torch.sigmoid(all_logits / self.temperature)
            loss = nn.functional.binary_cross_entropy(
                calibrated_probs,
                val_relevance_labels
            )
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        print(f"Calibrated relevance temperature: {self.temperature.item():.3f}")
```

### Feature-Specific Trust for Visual Regions

```python
class RegionwiseTrustTracker:
    """Track trust per visual region in ARR-COC"""

    def __init__(self, num_regions):
        self.num_regions = num_regions
        self.trust_scores = torch.ones(num_regions) * 0.5  # Initialize neutral
        self.performance_history = []

    def update_region_trust(self, region_idx, predicted_relevance, actual_relevance):
        """Update trust for specific region based on prediction quality"""

        # Compute prediction error
        error = abs(predicted_relevance - actual_relevance)

        # Update trust with exponential moving average
        alpha = 0.1
        new_trust = 1.0 - error  # High error = low trust
        self.trust_scores[region_idx] = (
            (1 - alpha) * self.trust_scores[region_idx] +
            alpha * new_trust
        )

        # Record history
        self.performance_history.append({
            'region': region_idx,
            'error': error.item(),
            'trust': self.trust_scores[region_idx].item()
        })

    def get_trusted_regions(self, threshold=0.7):
        """Get regions with trust above threshold"""
        return torch.where(self.trust_scores > threshold)[0]

    def adjust_token_budget_by_trust(self, base_budgets):
        """Allocate more tokens to trusted regions"""
        # Scale budgets by trust scores
        trust_weighted_budgets = base_budgets * self.trust_scores

        # Renormalize to preserve total budget
        total_budget = base_budgets.sum()
        adjusted_budgets = trust_weighted_budgets * (total_budget / trust_weighted_budgets.sum())

        return adjusted_budgets.int()
```

---

## 7. Research Papers & Resources (2024-2025)

### Key Papers

1. **[The Trust Calibration Maturity Model for AI Systems](https://arxiv.org/pdf/2503.15511)** (arXiv 2025)
   - Framework for assessing trust calibration maturity
   - Five-level model from ad-hoc to optimized trust calibration

2. **[The Key to Calibrating Trust and Optimal Decision Making with AI](https://academic.oup.com/pnasnexus/article/4/5/pgaf133/8118889)** (PNAS Nexus 2025)
   - Effect of confidence and explanation on trust calibration
   - Human subject studies on AI-assisted decision making

3. **[Trust Calibration for Joint Human/AI Decision-Making](https://dl.acm.org/doi/10.1007/978-3-031-93412-4_6)** (ACM 2025)
   - Trust calibration in dynamic, collaborative settings
   - Aligning trust with system trustworthiness

4. **[Dynamic Trust Calibration Using Contextual Bandits](https://arxiv.org/abs/2509.23497)** (arXiv 2025)
   - Novel contextual bandit approach to trust calibration
   - 10-38% improvement in decision quality across three datasets

5. **[Feature-Specific Trust Calibration in Physical AI Systems](https://aisel.aisnet.org/icis2025/user_behav/user_behav/14/)** (ICIS 2025)
   - Granular trust assessment per capability
   - Temporal dynamics of trust in physical AI systems

6. **[Trust in AI: Progress, Challenges, and Future Directions](https://www.nature.com/articles/s41599-024-04044-8)** (Nature 2024)
   - Comprehensive review of trust in AI systems
   - 172 citations, covers adaptive trust calibration methods

7. **[Calibrating Language Models with Adaptive Temperature Scaling](https://aclanthology.org/2024.emnlp-main.1007.pdf)** (EMNLP 2024)
   - Input-dependent temperature scaling for LLMs
   - 31% improvement over fixed temperature scaling

8. **[On Calibration of Modern Neural Networks](https://dl.acm.org/doi/10.5555/3305381.3305518)** (ICML 2017)
   - Foundational paper on neural network calibration
   - Introduced temperature scaling, ECE metrics

### Implementation Resources

- **[PyTorch Metrics - Calibration Error](https://lightning.ai/docs/torchmetrics/stable/classification/calibration_error.html)** - Official torchmetrics implementation of ECE
- **[Scikit-learn Temperature Scaling Issue](https://github.com/scikit-learn/scikit-learn/issues/28574)** - Discussion of implementing calibration in sklearn
- **[Expected Calibration Error (ECE) Tutorial](https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/)** - Visual explanation with Python code

### Key Metrics

| Metric | What It Measures | Good Values | Use Case |
|--------|------------------|-------------|----------|
| **ECE** (Expected Calibration Error) | Gap between confidence and accuracy | 0-0.05 (well-calibrated) | Primary calibration metric |
| **Brier Score** | Calibration + discrimination | 0-0.2 (good) | Combined performance |
| **MCE** (Max Calibration Error) | Worst-case calibration gap | 0-0.10 (acceptable) | Robust calibration |
| **Trust Score** | Dynamic trust level | 0.7-1.0 (high trust) | Runtime monitoring |

---

## 8. Summary and Best Practices

### When to Use Each Method

1. **Temperature Scaling**
   - **Use when**: Limited validation data, need simple method
   - **Pros**: 1 parameter, fast, rarely overfits
   - **Cons**: Global scaling, doesn't adapt to input

2. **Adaptive Temperature Scaling**
   - **Use when**: Input-dependent calibration needed, have enough data
   - **Pros**: More flexible, better performance on LLMs
   - **Cons**: More parameters, requires training

3. **Platt Scaling**
   - **Use when**: Binary classification, need per-class calibration
   - **Pros**: Well-studied, interpretable
   - **Cons**: Can overfit with small validation sets

4. **Ensemble Calibration**
   - **Use when**: Have computational budget, want uncertainty estimates
   - **Pros**: Better calibration, natural uncertainty quantification
   - **Cons**: Expensive, requires training multiple models

### Implementation Checklist

For trust-calibrated AI systems:

- [ ] **Measure baseline calibration** (ECE, Brier) on validation set
- [ ] **Choose calibration method** based on data size and requirements
- [ ] **Calibrate on held-out validation set** (never train set)
- [ ] **Monitor calibration during deployment** (detect distribution shift)
- [ ] **Implement feature-specific trust** for multi-capability systems
- [ ] **Provide uncertainty estimates** alongside predictions
- [ ] **Log trust violations** and retrain calibration periodically
- [ ] **Test with human users** to verify trust aligns with performance

### Key Takeaways

1. **Trust calibration is essential** for human-AI collaboration in high-stakes domains
2. **Neural networks are overconfident** - post-hoc calibration is necessary
3. **Feature-specific trust** recognizes heterogeneous capability reliability
4. **Dynamic trust adjustment** adapts to changing contexts and performance
5. **Temperature scaling** is simple and effective for most cases
6. **Monitor continuously** - calibration drifts over time
7. **Combine calibration + uncertainty** for honest confidence estimates

Trust calibration transforms AI from "black box oracle" to **honest collaborator** that knows and communicates its limitations.

---

## Sources

**Source Documents:**
- None (pure web research expansion)

**Web Research (2024-2025):**

**Trust Calibration Frameworks:**
- [The Trust Calibration Maturity Model (arXiv 2025)](https://arxiv.org/pdf/2503.15511) - Maturity framework for trust calibration
- [The Key to Calibrating Trust and Optimal Decision Making (PNAS Nexus 2025)](https://academic.oup.com/pnasnexus/article/4/5/pgaf133/8118889) - Human studies on trust calibration
- [Trust Calibration for Joint Human/AI Decision-Making (ACM 2025)](https://dl.acm.org/doi/10.1007/978-3-031-93412-4_6) - Dynamic trust in collaborative settings

**Dynamic Trust Methods:**
- [Dynamic Trust Calibration Using Contextual Bandits (arXiv:2509.23497, 2025)](https://arxiv.org/abs/2509.23497) - Contextual bandit approach to trust calibration
- [Dynamic Trust Scoring in IAM (Identity Management Institute 2025)](https://identitymanagementinstitute.org/dynamic-trust-scoring-in-iam/) - Real-time trust scoring methods

**Feature-Specific Trust:**
- [Feature-Specific Trust Calibration in Physical AI Systems (ICIS 2025)](https://aisel.aisnet.org/icis2025/user_behav/user_behav/14/) - Granular trust per capability in physical AI
- [A Systematic Literature Review of User Trust in AI-Enabled Systems (Taylor & Francis 2024)](https://www.tandfonline.com/doi/full/10.1080/10447318.2022.2138826) - User characteristics influencing trust

**Uncertainty Quantification:**
- [Neural Networks Calibration via Learning Uncertainty-Error (arXiv 2025)](https://arxiv.org/pdf/2505.22803) - CLUE method for alignment of uncertainty and error
- [Neural Parameter Calibration and Uncertainty Quantification (PLOS ONE 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0306704) - Uncertainty quantification for contagion models

**Calibration Methods:**
- [On Calibration of Modern Neural Networks (ICML 2017)](https://dl.acm.org/doi/10.5555/3305381.3305518) - Foundational temperature scaling paper
- [Calibrating Language Models with Adaptive Temperature Scaling (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1007.pdf) - Adaptive temperature for LLMs
- [Temperature Calibration for Higher Confidence (Cryptology ePrint 2024)](https://eprint.iacr.org/2024/071.pdf) - Temperature scaling applications

**Calibration Metrics:**
- [Expected Calibration Error Tutorial (Towards Data Science 2023)](https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/) - Visual ECE explanation
- [Calibration Error — PyTorch-Metrics Documentation](https://lightning.ai/docs/torchmetrics/stable/classification/calibration_error.html) - Official torchmetrics implementation
- [Achieving Well-Informed Decision-Making in Drug Discovery (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11881400/) - Brier score decomposition

**Comprehensive Reviews:**
- [Trust in AI: Progress, Challenges, and Future Directions (Nature 2024)](https://www.nature.com/articles/s41599-024-04044-8) - Comprehensive trust in AI review (172 citations)
- [From Trust in Automation to Trust in AI in Healthcare (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12562135/) - 30-year review of human-machine trust

**Additional References:**
- [Real-Time AI Governance: Monitoring, Compliance (Relyance AI 2025)](https://www.relyance.ai/blog/ai-governance) - Continuous monitoring for AI systems
- [AI TRiSM (Proofpoint 2024)](https://www.proofpoint.com/us/threat-reference/ai-trism) - AI Trust, Risk, and Security Management framework
- [AI Monitoring Explained (Cribl 2025)](https://cribl.io/glossary/ai-monitoring/) - Real-time AI system monitoring
- [Calibration Techniques in Deep Neural Networks (Heartbeat 2023)](https://heartbeat.comet.ml/calibration-techniques-in-deep-neural-networks-55ad76fea58b) - Overview of calibration methods

All research accessed 2025-01-31.
