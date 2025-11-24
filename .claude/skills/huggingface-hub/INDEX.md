# HuggingFace Hub Documentation Index

Quick navigation index for HuggingFace Hub documentation skill.

## 📖 Core Concepts
Start here for understanding the Hub:
- **[Hub Overview](concepts/hub-overview.md)** - What is HuggingFace Hub and what can you do with it

## 🤖 Models
Working with ML models on the Hub:

### Core Operations
- **[The Model Hub](models/the-hub.md)** - Browse and discover models
- **[Uploading Models](models/uploading.md)** - How to upload your models
- **[Downloading Models](models/downloading.md)** - How to download and use models
- **[Model Cards](models/model-cards.md)** - Documentation and metadata for models
- **[Model Inference](models/inference.md)** - Run inference on models via API
- **[Model Widgets](models/widgets.md)** - Interactive demos for models
- **[Gated Models](models/gated.md)** - Access control for models

### Library Integrations
- **[Transformers](models/libraries/transformers.md)** - Hugging Face Transformers integration
- **[Diffusers](models/libraries/diffusers.md)** - Diffusion models integration
- **[PEFT](models/libraries/peft.md)** - Parameter-Efficient Fine-Tuning integration

## 📊 Datasets
Working with datasets on the Hub:

### Core Operations
- **[Overview](datasets/overview.md)** - Introduction to datasets on the Hub
- **[Uploading Datasets](datasets/uploading.md)** - How to upload your datasets
- **[Downloading Datasets](datasets/downloading.md)** - How to download and use datasets
- **[Dataset Cards](datasets/dataset-cards.md)** - Documentation and metadata for datasets
- **[Data Studio](datasets/data-studio.md)** - Explore datasets in your browser
- **[Gated Datasets](datasets/gated.md)** - Access control for datasets

### Data Tool Integrations
- **[DuckDB](datasets/integrations/duckdb.md)** - Query datasets with SQL
- **[Polars](datasets/integrations/polars.md)** - Fast dataframe operations
- **[Pandas](datasets/integrations/pandas.md)** - Classic dataframe operations

## 🚀 Spaces
Interactive demo apps and applications:
- **[Overview](spaces/overview.md)** - Introduction to Spaces
- **[Gradio Spaces](spaces/gradio.md)** - Build Gradio apps
- **[Docker Spaces](spaces/docker.md)** - Custom Docker deployments
- **[GPU Upgrades](spaces/gpu-upgrades.md)** - Add GPU acceleration
- **[ZeroGPU](spaces/zerogpu.md)** - Dynamic GPU allocation
- **[Configuration Reference](spaces/configuration.md)** - Space settings and config

## 📦 Repositories
Managing your Hub repositories:
- **[Getting Started](repositories/getting-started.md)** - Basics of Hub repositories
- **[Repository Settings](repositories/settings.md)** - Configure your repositories
- **[Pull Requests & Discussions](repositories/pull-requests.md)** - Collaborate on repositories
- **[Webhooks](repositories/webhooks.md)** - Automate workflows with webhooks

## 🏢 Enterprise & Billing
Enterprise features and account management:
- **[Enterprise Overview & Security](enterprise/overview-and-security.md)** - Enterprise features and advanced security
- **[Audit Logs](enterprise/audit-logs.md)** - Track organization activity
- **[Billing](enterprise/billing.md)** - Manage subscriptions and payments
- **[PRO Plan](enterprise/pro-plan.md)** - Individual PRO subscription features
- **[Rate Limits](enterprise/rate-limits.md)** - API and usage quotas

## 🔧 Advanced Topics
Advanced features and integrations:
- **[API Endpoints](advanced/api-endpoints.md)** - Programmatic Hub access
- **[Security](advanced/security.md)** - Tokens, access control, GPG signing
- **[Organizations](advanced/organizations.md)** - Team accounts and collaboration
- **[Agents](advanced/agents.md)** - AI agents on the Hub

---

## Quick Reference by Task

### "I want to upload a model"
→ [models/uploading.md](models/uploading.md)

### "I want to use a model in my code"
→ [models/downloading.md](models/downloading.md) + [models/libraries/transformers.md](models/libraries/transformers.md)

### "I want to upload a dataset"
→ [datasets/uploading.md](datasets/uploading.md)

### "I want to query datasets with SQL"
→ [datasets/integrations/duckdb.md](datasets/integrations/duckdb.md)

### "I want to create an interactive demo"
→ [spaces/overview.md](spaces/overview.md) + [spaces/gradio.md](spaces/gradio.md)

### "I want to run inference without downloading"
→ [models/inference.md](models/inference.md)

### "I want to control who accesses my model/dataset"
→ [models/gated.md](models/gated.md) or [datasets/gated.md](datasets/gated.md)

### "I need to use the API programmatically"
→ [advanced/api-endpoints.md](advanced/api-endpoints.md)

### "I want to set up an organization"
→ [advanced/organizations.md](advanced/organizations.md)

---

## File Organization

```
.claude/skills/huggingface-hub/
├── INDEX.md (this file)
├── skill.md (instructions for Claude)
├── concepts/
│   └── hub-overview.md
├── models/
│   ├── the-hub.md
│   ├── uploading.md
│   ├── downloading.md
│   ├── model-cards.md
│   ├── inference.md
│   ├── widgets.md
│   ├── gated.md
│   └── libraries/
│       ├── transformers.md
│       ├── diffusers.md
│       └── peft.md
├── datasets/
│   ├── overview.md
│   ├── uploading.md
│   ├── downloading.md
│   ├── dataset-cards.md
│   ├── data-studio.md
│   ├── gated.md
│   └── integrations/
│       ├── duckdb.md
│       ├── polars.md
│       └── pandas.md
├── spaces/
│   ├── overview.md
│   ├── gradio.md
│   ├── docker.md
│   ├── gpu-upgrades.md
│   ├── zerogpu.md
│   └── configuration.md
├── repositories/
│   ├── getting-started.md
│   ├── settings.md
│   ├── pull-requests.md
│   └── webhooks.md
├── enterprise/
│   ├── overview-and-security.md
│   ├── audit-logs.md
│   ├── billing.md
│   ├── pro-plan.md
│   └── rate-limits.md
└── advanced/
    ├── api-endpoints.md
    ├── security.md
    ├── organizations.md
    └── agents.md
```

## Notes

- This skill contains core HuggingFace Hub documentation
- For detailed library-specific docs (transformers, datasets, etc.), use those libraries' documentation
- Files are extracted from official HuggingFace Hub docs
- Source: https://huggingface.co/docs/hub/
