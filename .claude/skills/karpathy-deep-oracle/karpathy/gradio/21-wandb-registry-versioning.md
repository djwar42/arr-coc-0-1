# W&B Registry: Model Versioning & Lifecycle Management

**Category:** Gradio Integration & Production ML
**Context:** Advanced W&B features for production model deployment
**Skill Level:** Advanced
**Prerequisites:** [gradio/10-wandb-basics.md](gradio/10-wandb-basics.md), [gradio/11-wandb-artifacts.md](gradio/11-wandb-artifacts.md)

---

## Overview

W&B Registry is a central repository for managing and versioning ML artifacts (models, datasets) across your organization. It extends W&B Artifacts with organization-level governance, lifecycle management, and production deployment workflows.

**Key Distinction:**
- **W&B Artifacts**: Project-level versioning and tracking (private to project)
- **W&B Registry**: Organization-level curation and sharing (shared across teams)

From [W&B Registry Documentation](https://docs.wandb.ai/models/registry) (accessed 2025-01-31):
> "W&B Registry is a curated central repository of W&B Artifact versions within your organization. Users who have permission can download and use artifacts, share, and collaboratively manage the lifecycle of all artifacts, regardless of the team that user belongs to."

**Use Registry when you need:**
- Production model deployment workflows
- Cross-team artifact sharing
- Model lifecycle stages (dev → staging → production)
- Governance and compliance tracking
- Automated CI/CD triggers

**Use Artifacts when you need:**
- Experiment-level tracking
- Dataset versioning within a project
- Model checkpoint management during training
- Private work before publishing

---

## Section 1: Registry Fundamentals (130 lines)

### Registry vs Artifacts Architecture

**Artifacts (Project-Level):**
```python
import wandb

run = wandb.init(project="my-experiments", entity="my-team")

# Log artifact to project
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")
run.log_artifact(artifact)  # Private to project
```

**Registry (Organization-Level):**
```python
import wandb

run = wandb.init(project="my-experiments", entity="my-team")

# Log artifact
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")
logged_artifact = run.log_artifact(artifact)

# Link to registry (publish to organization)
REGISTRY_NAME = "model"  # Core registry
COLLECTION_NAME = "production-classifier"

run.link_artifact(
    artifact=logged_artifact,
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

From [W&B Registry Overview](https://docs.wandb.ai/models/registry) (accessed 2025-01-31):
> "The term 'link' refers to pointers that connect where W&B stores the artifact and where the artifact is accessible in the registry. W&B does not duplicate artifacts when you link an artifact to a collection."

**Key Concepts:**

1. **Registry**: Organization-wide hub (e.g., "Models", "Datasets")
2. **Collection**: Task-specific grouping (e.g., "image-classifier", "vqa-model")
3. **Artifact Version**: Specific snapshot (v0, v1, v2, etc.)
4. **Alias**: Named pointer (latest, production, staging, candidate)

### Core Registries

W&B provides two default registries:

**1. Model Registry** (`wandb-registry-model`)
```python
# Link model artifact
artifact = wandb.Artifact("resnet50-finetuned", type="model")
artifact.add_file("model.pth")
logged_artifact = run.log_artifact(artifact)

run.link_artifact(
    artifact=logged_artifact,
    target_path="wandb-registry-model/image-classifier"
)
```

**2. Dataset Registry** (`wandb-registry-dataset`)
```python
# Link dataset artifact
artifact = wandb.Artifact("training-data-v2", type="dataset")
artifact.add_dir("data/")
logged_artifact = run.log_artifact(artifact)

run.link_artifact(
    artifact=logged_artifact,
    target_path="wandb-registry-dataset/imagenet-subset"
)
```

### Versioning Strategy

**Automatic Versioning:**
```python
# First link creates v0
run.link_artifact(artifact, target_path="wandb-registry-model/classifier")
# → Creates: wandb-registry-model/classifier:v0

# Second link creates v1
run.link_artifact(artifact, target_path="wandb-registry-model/classifier")
# → Creates: wandb-registry-model/classifier:v1
```

**Semantic Versioning with Aliases:**
```python
# Development version
artifact.aliases = ["dev", "latest"]
run.link_artifact(artifact, target_path="wandb-registry-model/classifier")

# Staging version
artifact.aliases = ["staging", "v1.0.0-rc1"]
run.link_artifact(artifact, target_path="wandb-registry-model/classifier")

# Production version
artifact.aliases = ["production", "v1.0.0", "latest"]
run.link_artifact(artifact, target_path="wandb-registry-model/classifier")
```

### Model Lifecycle Stages

**Typical Flow:**
```
Training → Development → Staging → Production
   ↓           ↓           ↓          ↓
  v0         :dev      :staging  :production
```

**Implementation:**
```python
import wandb

def promote_model(version_index, new_alias):
    """Promote model through lifecycle stages"""
    run = wandb.init(project="model-lifecycle")

    # Download specific version
    artifact = run.use_artifact(
        f"wandb-registry-model/classifier:v{version_index}"
    )

    # Add new alias (promotes to new stage)
    artifact.aliases.append(new_alias)

    # Re-link with updated aliases
    run.link_artifact(
        artifact=artifact,
        target_path="wandb-registry-model/classifier"
    )

    run.finish()

# Example: Promote v3 to production
promote_model(version_index=3, new_alias="production")
```

### Model Metadata and Tags

From [W&B Registry Documentation](https://docs.wandb.ai/models/registry/organize-with-tags) (accessed 2025-01-31):
> "Use tags to organize collections or artifact versions within collections. You can add, remove, edit tags with the Python SDK or W&B App UI."

**Adding Metadata:**
```python
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")

# Add metadata
artifact.metadata = {
    "framework": "PyTorch",
    "architecture": "ResNet50",
    "training_acc": 0.95,
    "val_acc": 0.92,
    "dataset": "ImageNet-1K",
    "batch_size": 128,
    "learning_rate": 0.001
}

logged_artifact = run.log_artifact(artifact)
```

**Organizing with Tags:**
```python
# Tag for filtering and organization
artifact.metadata["tags"] = [
    "computer-vision",
    "classification",
    "production-ready"
]
```

---

## Section 2: Production Workflows (150 lines)

### Development → Staging → Production Pipeline

**Complete Workflow:**
```python
import wandb
from pathlib import Path

class ModelRegistry:
    """Manage model lifecycle through registry"""

    def __init__(self, collection_name="my-model"):
        self.collection_name = collection_name
        self.registry_path = f"wandb-registry-model/{collection_name}"

    def publish_dev(self, model_path, metrics):
        """Publish development version"""
        run = wandb.init(
            project="model-dev",
            job_type="publish-dev"
        )

        # Create artifact
        artifact = wandb.Artifact(
            name=f"{self.collection_name}-dev",
            type="model",
            metadata={
                "stage": "development",
                "metrics": metrics,
                "git_commit": get_git_commit()
            }
        )
        artifact.add_file(model_path)

        # Log and link with 'dev' alias
        logged_artifact = run.log_artifact(artifact)
        artifact.aliases = ["dev", "latest"]

        run.link_artifact(
            artifact=logged_artifact,
            target_path=self.registry_path
        )

        run.finish()
        print(f"Published to development: {self.registry_path}:dev")

    def promote_to_staging(self, version_or_alias="dev"):
        """Promote model to staging"""
        run = wandb.init(
            project="model-staging",
            job_type="promote-staging"
        )

        # Download from registry
        artifact = run.use_artifact(
            f"{self.registry_path}:{version_or_alias}"
        )

        # Update aliases for staging
        artifact.aliases = ["staging", "v-staging-latest"]

        # Re-link with staging alias
        run.link_artifact(
            artifact=artifact,
            target_path=self.registry_path
        )

        run.finish()
        print(f"Promoted to staging: {self.registry_path}:staging")

    def promote_to_production(self, version_or_alias="staging",
                             version_tag=None):
        """Promote model to production with optional version tag"""
        run = wandb.init(
            project="model-production",
            job_type="promote-production"
        )

        # Download from registry
        artifact = run.use_artifact(
            f"{self.registry_path}:{version_or_alias}"
        )

        # Production aliases
        aliases = ["production"]
        if version_tag:
            aliases.append(version_tag)  # e.g., "v1.0.0"

        artifact.aliases = aliases

        # Re-link with production alias
        run.link_artifact(
            artifact=artifact,
            target_path=self.registry_path
        )

        run.finish()
        print(f"Promoted to production: {self.registry_path}:production")

    def rollback(self, to_version):
        """Rollback production to specific version"""
        run = wandb.init(
            project="model-production",
            job_type="rollback"
        )

        # Get specific version
        artifact = run.use_artifact(
            f"{self.registry_path}:v{to_version}"
        )

        # Set as production
        artifact.aliases = ["production", "rollback"]

        run.link_artifact(
            artifact=artifact,
            target_path=self.registry_path
        )

        run.finish()
        print(f"Rolled back production to v{to_version}")

# Usage
registry = ModelRegistry(collection_name="resnet-classifier")

# 1. Publish development model
registry.publish_dev(
    model_path="models/resnet_epoch50.pt",
    metrics={"val_acc": 0.92, "val_loss": 0.25}
)

# 2. Promote to staging (after validation)
registry.promote_to_staging(version_or_alias="dev")

# 3. Promote to production (after staging tests)
registry.promote_to_production(
    version_or_alias="staging",
    version_tag="v1.2.0"
)

# 4. Emergency rollback if needed
registry.rollback(to_version=5)
```

### Approval Workflows

**Manual Approval Pattern:**
```python
import wandb

def request_production_approval(model_version, metrics):
    """Create approval request in W&B"""
    run = wandb.init(
        project="model-approvals",
        job_type="approval-request"
    )

    # Log approval request
    run.log({
        "requested_version": model_version,
        "requesting_user": wandb.api.viewer()["username"],
        "metrics": metrics,
        "timestamp": wandb.util.generate_id()
    })

    # Link to approval collection
    artifact = run.use_artifact(
        f"wandb-registry-model/classifier:{model_version}"
    )

    artifact.metadata["approval_status"] = "pending"
    artifact.metadata["approval_requested_at"] = run.start_time

    run.link_artifact(
        artifact=artifact,
        target_path="wandb-registry-model/classifier"
    )

    run.finish()
    print(f"Approval requested for version {model_version}")

def approve_for_production(model_version, approver):
    """Approve model for production deployment"""
    run = wandb.init(
        project="model-approvals",
        job_type="approval-grant"
    )

    artifact = run.use_artifact(
        f"wandb-registry-model/classifier:{model_version}"
    )

    # Update approval metadata
    artifact.metadata["approval_status"] = "approved"
    artifact.metadata["approved_by"] = approver
    artifact.metadata["approved_at"] = wandb.util.generate_id()

    # Promote to production
    artifact.aliases = ["production", "approved"]

    run.link_artifact(
        artifact=artifact,
        target_path="wandb-registry-model/classifier"
    )

    run.finish()
    print(f"Model {model_version} approved and promoted to production")
```

### A/B Deployment Patterns

**Blue-Green Deployment:**
```python
def deploy_blue_green(new_model_version, traffic_split=0.1):
    """Deploy new model alongside current production"""
    run = wandb.init(
        project="model-deployment",
        job_type="blue-green-deploy"
    )

    # Current production (blue)
    blue_artifact = run.use_artifact(
        "wandb-registry-model/classifier:production"
    )

    # New candidate (green)
    green_artifact = run.use_artifact(
        f"wandb-registry-model/classifier:v{new_model_version}"
    )

    # Set aliases for routing
    blue_artifact.aliases = ["production-blue", f"traffic-{100-traffic_split*100:.0f}"]
    green_artifact.aliases = ["production-green", f"traffic-{traffic_split*100:.0f}"]

    # Link both
    run.link_artifact(blue_artifact, target_path="wandb-registry-model/classifier")
    run.link_artifact(green_artifact, target_path="wandb-registry-model/classifier")

    # Log deployment config
    run.config.update({
        "blue_version": blue_artifact.version,
        "green_version": green_artifact.version,
        "traffic_split": traffic_split
    })

    run.finish()
    print(f"Deployed blue-green: {traffic_split*100}% to v{new_model_version}")
```

### Model Performance Monitoring in Production

From [W&B Production Monitoring Best Practices](https://wandb.ai/site/articles/intro-to-mlops-data-and-model-versioning) (accessed 2025-01-31):
> "ML models are versioned in a special repository, called a model registry, for storing and managing different model versions throughout the entire model lifecycle."

**Production Metrics Tracking:**
```python
class ProductionModelMonitor:
    """Monitor production model performance"""

    def __init__(self, registry_path):
        self.registry_path = registry_path
        self.run = wandb.init(
            project="production-monitoring",
            job_type="monitoring"
        )

        # Load production model metadata
        self.artifact = self.run.use_artifact(
            f"{registry_path}:production"
        )

    def log_inference(self, prediction, latency_ms, confidence):
        """Log single inference"""
        self.run.log({
            "prediction": prediction,
            "latency_ms": latency_ms,
            "confidence": confidence,
            "timestamp": wandb.util.generate_id()
        })

    def log_batch_metrics(self, batch_metrics):
        """Log batch inference metrics"""
        self.run.log({
            "batch_size": batch_metrics["batch_size"],
            "avg_latency_ms": batch_metrics["avg_latency"],
            "p95_latency_ms": batch_metrics["p95_latency"],
            "p99_latency_ms": batch_metrics["p99_latency"],
            "avg_confidence": batch_metrics["avg_confidence"],
            "throughput_qps": batch_metrics["qps"]
        })

    def check_drift(self, prediction_distribution):
        """Check for prediction drift"""
        # Compare with training distribution
        training_dist = self.artifact.metadata.get("training_distribution")

        drift_score = calculate_drift(prediction_distribution, training_dist)

        self.run.log({
            "drift_score": drift_score,
            "alert": drift_score > 0.2  # Threshold
        })

        if drift_score > 0.2:
            self.trigger_retraining_alert()

    def trigger_retraining_alert(self):
        """Trigger alert for model retraining"""
        self.run.alert(
            title="Model Drift Detected",
            text=f"Production model in {self.registry_path} showing drift",
            level=wandb.AlertLevel.WARN
        )

# Usage in production service
monitor = ProductionModelMonitor("wandb-registry-model/classifier")

# Log inferences
monitor.log_inference(
    prediction="cat",
    latency_ms=15.3,
    confidence=0.95
)

# Check for drift periodically
monitor.check_drift(current_predictions)
```

---

## Section 3: Model Cards & Documentation (120 lines)

### Creating Model Cards

From [W&B Model Cards Documentation](https://docs.wandb.ai/models/registry/registry_cards) (accessed 2025-01-31):
> "Annotate collections with descriptions, documentation, and metadata to help users understand the artifacts in your registry."

**Complete Model Card Example:**
```python
import wandb

def create_model_card(collection_name, model_metadata):
    """Create comprehensive model card"""
    run = wandb.init(
        project="model-documentation",
        job_type="create-model-card"
    )

    artifact = wandb.Artifact(
        name=f"{collection_name}-card",
        type="model",
        description=model_metadata["description"]
    )

    # Core metadata
    artifact.metadata = {
        # Model information
        "model_name": model_metadata["name"],
        "version": model_metadata["version"],
        "architecture": model_metadata["architecture"],
        "framework": model_metadata["framework"],
        "task": model_metadata["task"],

        # Performance metrics
        "metrics": {
            "accuracy": model_metadata["accuracy"],
            "precision": model_metadata["precision"],
            "recall": model_metadata["recall"],
            "f1_score": model_metadata["f1_score"]
        },

        # Training information
        "training": {
            "dataset": model_metadata["training_dataset"],
            "dataset_size": model_metadata["dataset_size"],
            "training_date": model_metadata["training_date"],
            "training_duration_hours": model_metadata["training_hours"],
            "hyperparameters": model_metadata["hyperparameters"]
        },

        # Deployment information
        "deployment": {
            "inference_latency_ms": model_metadata["latency"],
            "memory_requirements_gb": model_metadata["memory_gb"],
            "compute_requirements": model_metadata["compute"],
            "compatible_hardware": model_metadata["hardware"]
        },

        # Governance
        "governance": {
            "owner": model_metadata["owner"],
            "team": model_metadata["team"],
            "contact": model_metadata["contact"],
            "license": model_metadata["license"]
        }
    }

    # Add model files
    artifact.add_file(model_metadata["model_path"])

    # Add documentation
    if "readme_path" in model_metadata:
        artifact.add_file(model_metadata["readme_path"], name="README.md")

    # Link to registry
    logged_artifact = run.log_artifact(artifact)
    run.link_artifact(
        artifact=logged_artifact,
        target_path=f"wandb-registry-model/{collection_name}"
    )

    run.finish()
    return artifact

# Example usage
model_card_metadata = {
    "name": "ResNet50 Image Classifier",
    "version": "v1.0.0",
    "description": "Fine-tuned ResNet50 for multi-class image classification",
    "architecture": "ResNet50",
    "framework": "PyTorch 2.0",
    "task": "image-classification",

    "accuracy": 0.94,
    "precision": 0.93,
    "recall": 0.92,
    "f1_score": 0.925,

    "training_dataset": "ImageNet-1K subset (100K images)",
    "dataset_size": "100,000 images",
    "training_date": "2025-01-15",
    "training_hours": 48,
    "hyperparameters": {
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "epochs": 50
    },

    "latency": 15.3,
    "memory_gb": 2.5,
    "compute": "1x NVIDIA T4",
    "hardware": ["GPU", "CPU (slower)"],

    "owner": "ml-team@company.com",
    "team": "Computer Vision",
    "contact": "cv-team@company.com",
    "license": "MIT",

    "model_path": "models/resnet50_v1.pth",
    "readme_path": "docs/MODEL_README.md"
}

create_model_card("image-classifier", model_card_metadata)
```

### Performance Metrics Documentation

**Detailed Metrics Tracking:**
```python
def document_performance_metrics(collection_name, evaluation_results):
    """Document comprehensive performance metrics"""
    run = wandb.init(
        project="model-evaluation",
        job_type="document-metrics"
    )

    # Load model from registry
    artifact = run.use_artifact(
        f"wandb-registry-model/{collection_name}:latest"
    )

    # Add evaluation metrics
    artifact.metadata["evaluation"] = {
        # Overall metrics
        "overall_accuracy": evaluation_results["accuracy"],
        "overall_f1": evaluation_results["f1"],

        # Per-class metrics
        "per_class_metrics": evaluation_results["per_class"],

        # Confusion matrix
        "confusion_matrix": evaluation_results["confusion_matrix"],

        # Error analysis
        "common_errors": evaluation_results["error_analysis"],
        "failure_cases": evaluation_results["failure_cases"],

        # Robustness testing
        "adversarial_accuracy": evaluation_results.get("adversarial_acc"),
        "out_of_distribution_performance": evaluation_results.get("ood_perf")
    }

    # Log visualizations
    if "confusion_matrix_plot" in evaluation_results:
        wandb.log({"confusion_matrix": evaluation_results["confusion_matrix_plot"]})

    # Re-link with updated metadata
    run.link_artifact(
        artifact=artifact,
        target_path=f"wandb-registry-model/{collection_name}"
    )

    run.finish()
```

### Training Data Provenance

**Dataset Lineage Tracking:**
```python
def link_training_data_provenance(model_collection, dataset_collection):
    """Link model to its training dataset"""
    run = wandb.init(
        project="data-lineage",
        job_type="link-provenance"
    )

    # Get model artifact
    model_artifact = run.use_artifact(
        f"wandb-registry-model/{model_collection}:latest"
    )

    # Get dataset artifact
    dataset_artifact = run.use_artifact(
        f"wandb-registry-dataset/{dataset_collection}:latest"
    )

    # Link dataset to model metadata
    model_artifact.metadata["training_data"] = {
        "dataset_collection": dataset_collection,
        "dataset_version": dataset_artifact.version,
        "dataset_size": dataset_artifact.metadata["size"],
        "dataset_hash": dataset_artifact.digest,
        "preprocessing": dataset_artifact.metadata.get("preprocessing"),
        "augmentations": dataset_artifact.metadata.get("augmentations")
    }

    # Re-link model
    run.link_artifact(
        artifact=model_artifact,
        target_path=f"wandb-registry-model/{model_collection}"
    )

    run.finish()
    print(f"Linked {model_collection} to {dataset_collection}")
```

### Known Limitations and Biases

**Documenting Model Limitations:**
```python
def document_limitations(collection_name, limitations):
    """Document known limitations and biases"""
    run = wandb.init(
        project="model-documentation",
        job_type="document-limitations"
    )

    artifact = run.use_artifact(
        f"wandb-registry-model/{collection_name}:latest"
    )

    artifact.metadata["limitations"] = {
        # Performance limitations
        "performance": {
            "low_accuracy_classes": limitations["weak_classes"],
            "edge_cases": limitations["edge_cases"],
            "minimum_confidence_threshold": limitations["min_confidence"]
        },

        # Data biases
        "biases": {
            "training_data_bias": limitations["data_bias"],
            "demographic_bias": limitations.get("demographic_bias"),
            "geographical_bias": limitations.get("geo_bias")
        },

        # Technical constraints
        "constraints": {
            "max_input_size": limitations["max_input_size"],
            "supported_formats": limitations["supported_formats"],
            "requires_preprocessing": limitations["preprocessing_required"]
        },

        # Ethical considerations
        "ethical": {
            "not_suitable_for": limitations["prohibited_uses"],
            "requires_human_review": limitations["human_review_cases"],
            "fairness_considerations": limitations["fairness"]
        }
    }

    run.link_artifact(
        artifact=artifact,
        target_path=f"wandb-registry-model/{collection_name}"
    )

    run.finish()

# Example
limitations = {
    "weak_classes": ["rare_bird_species", "occluded_objects"],
    "edge_cases": ["low_light_conditions", "extreme_angles"],
    "min_confidence": 0.7,
    "data_bias": "Training data primarily from North America",
    "max_input_size": "1024x1024 pixels",
    "supported_formats": ["JPEG", "PNG"],
    "preprocessing_required": True,
    "prohibited_uses": ["medical_diagnosis", "legal_decisions"],
    "human_review_cases": ["confidence < 0.8", "ambiguous_predictions"],
    "fairness": "Model may underperform on underrepresented demographics"
}

document_limitations("image-classifier", limitations)
```

### Integration with HuggingFace Hub

From [W&B Integration Best Practices](https://wandb.ai/site/articles/intro-to-mlops-data-and-model-versioning) (accessed 2025-01-31):
> "Model registries integrate with external ML systems for deployment and monitoring."

**Syncing to HuggingFace:**
```python
from huggingface_hub import HfApi
import wandb

def sync_to_huggingface(wandb_collection, hf_repo_id):
    """Sync W&B Registry model to HuggingFace Hub"""
    # Download from W&B Registry
    run = wandb.init(project="hf-sync")
    artifact = run.use_artifact(
        f"wandb-registry-model/{wandb_collection}:production"
    )
    artifact_dir = artifact.download()

    # Upload to HuggingFace
    api = HfApi()
    api.upload_folder(
        folder_path=artifact_dir,
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message=f"Synced from W&B {wandb_collection}:{artifact.version}"
    )

    # Link HF model in W&B metadata
    artifact.metadata["huggingface_hub"] = {
        "repo_id": hf_repo_id,
        "url": f"https://huggingface.co/{hf_repo_id}",
        "synced_at": wandb.util.generate_id()
    }

    run.link_artifact(
        artifact=artifact,
        target_path=f"wandb-registry-model/{wandb_collection}"
    )

    run.finish()
    print(f"Synced to HuggingFace: {hf_repo_id}")

# Example
sync_to_huggingface(
    wandb_collection="text-classifier",
    hf_repo_id="myorg/bert-classifier-v1"
)
```

---

## Sources

**W&B Official Documentation:**
- [W&B Registry Overview](https://docs.wandb.ai/models/registry) - Core registry concepts and workflows (accessed 2025-01-31)
- [Model Registry Documentation](https://docs.wandb.ai/models/registry/model_registry) - Legacy model registry (accessed 2025-01-31)
- [Link Artifact Version to Registry](https://docs.wandb.ai/models/registry/link_version) - Linking workflow (accessed 2025-01-31)
- [Organize with Tags](https://docs.wandb.ai/models/registry/organize-with-tags) - Tagging and organization (accessed 2025-01-31)

**Web Research:**
- [What is an ML Model Registry?](https://wandb.ai/site/articles/what-is-an-ML-model-registry/) - W&B Blog (accessed 2025-01-31)
- [Intro to MLOps: Data and Model Versioning](https://wandb.ai/site/articles/intro-to-mlops-data-and-model-versioning/) - W&B Blog (accessed 2025-01-31)
- [W&B Registry Product Page](https://wandb.ai/site/registry/) - Product overview (accessed 2025-01-31)

**Video Tutorials:**
- [Getting started with Registry from Weights & Biases](https://www.youtube.com/watch?v=p4XkVOsjIeM) - Official tutorial (accessed 2025-01-31)
- [Using W&B Model Registry to Manage Models](https://www.youtube.com/watch?v=Uw_EOuSoGYQ) - Model management workflow (accessed 2025-01-31)

**Additional References:**
- [W&B GitHub Repository](https://github.com/wandb/wandb) - Open source SDK (accessed 2025-01-31)

---

## Cross-References

**Prerequisites:**
- [gradio/10-wandb-basics.md](gradio/10-wandb-basics.md) - W&B fundamentals
- [gradio/11-wandb-artifacts.md](gradio/11-wandb-artifacts.md) - Artifact basics

**Related Topics:**
- [gradio/17-wandb-weave-llm-tracking.md](gradio/17-wandb-weave-llm-tracking.md) - LLM observability (when created)
- [gradio/18-wandb-evaluations.md](gradio/18-wandb-evaluations.md) - Model evaluation (when created)
- [practical-implementation/17-wandb-production-monitoring.md](../practical-implementation/17-wandb-production-monitoring.md) - Production monitoring (when created)

**Advanced Topics:**
- [practical-implementation/20-wandb-artifacts-advanced.md](../practical-implementation/20-wandb-artifacts-advanced.md) - Advanced artifact patterns (when created)
- [practical-implementation/21-wandb-integration-cookbook.md](../practical-implementation/21-wandb-integration-cookbook.md) - Integration examples (when created)
