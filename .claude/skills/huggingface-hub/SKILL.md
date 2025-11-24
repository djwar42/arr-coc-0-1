---
name: huggingface-hub
description: Comprehensive HuggingFace Hub documentation including model/dataset upload and download, Spaces (Gradio/Docker), repository management, enterprise features (SSO, audit logs), API endpoints, and integrations (DuckDB, Polars, Pandas). Use when questions involve HuggingFace Hub, model deployment, dataset hosting, Spaces apps, or Hub API.
---

# HuggingFace Hub Documentation Skill

Reference documentation for HuggingFace Hub - models, datasets, spaces, and repository management.

## When to Use This Skill

Use this skill when the user asks about:
- **Uploading/downloading** models or datasets to/from HuggingFace Hub
- **Model inference** via HuggingFace API
- **Spaces** (creating interactive demos with Gradio/Docker)
- **Dataset operations** (Data Studio, DuckDB, Polars integrations)
- **Repository management** (settings, pull requests, webhooks)
- **Enterprise features** (SSO, audit logs, billing, rate limits)
- **API endpoints** and programmatic Hub access
- **Organizations** and team collaboration
- **Access control** (gated models/datasets)

## Quick Navigation

The skill is organized into 7 main categories:

1. **concepts/** - Hub overview and introduction
2. **models/** - Model upload, download, inference, cards, widgets
3. **datasets/** - Dataset upload, download, viewer, integrations
4. **spaces/** - Interactive apps (Gradio, Docker, GPU)
5. **repositories/** - Git-based repo management
6. **enterprise/** - Billing, security, audit logs, rate limits
7. **advanced/** - API, security, organizations, agents

**Always start with [INDEX.md](INDEX.md) for quick navigation links.**

## Usage Patterns

### Pattern 1: Direct Task Lookup
User asks: "How do I upload a model to HuggingFace?"
→ Read `models/uploading.md`
→ Provide relevant instructions

### Pattern 2: Integration Guidance
User asks: "How do I use transformers with HuggingFace Hub?"
→ Read `models/libraries/transformers.md`
→ Provide integration code examples

### Pattern 3: Dataset Workflows
User asks: "How do I query a HuggingFace dataset with SQL?"
→ Read `datasets/integrations/duckdb.md`
→ Explain DuckDB integration

### Pattern 4: Enterprise Features
User asks: "How do I set up audit logs?"
→ Read `enterprise/audit-logs.md`
→ Explain Enterprise feature setup

### Pattern 5: Index First Approach
User asks vague question about Hub
→ Read `INDEX.md` first to understand structure
→ Then read relevant specific docs

## File Reference Guide

### Models Workflow
```
concepts/hub-overview.md → What is the Hub?
models/the-hub.md → Browse models
models/uploading.md → Upload your model
models/downloading.md → Download and use
models/model-cards.md → Document your model
models/inference.md → Run inference via API
models/widgets.md → Add interactive demo
models/gated.md → Control access
```

### Datasets Workflow
```
datasets/overview.md → Introduction
datasets/uploading.md → Upload dataset
datasets/downloading.md → Download dataset
datasets/dataset-cards.md → Document dataset
datasets/data-studio.md → Explore in browser
datasets/integrations/duckdb.md → SQL queries
datasets/integrations/polars.md → Fast dataframes
datasets/integrations/pandas.md → Classic dataframes
```

### Spaces Workflow
```
spaces/overview.md → What are Spaces?
spaces/gradio.md → Build Gradio app
spaces/docker.md → Custom Docker container
spaces/gpu-upgrades.md → Add GPU
spaces/zerogpu.md → Dynamic GPU allocation
spaces/configuration.md → Configure settings
```

### Repository Management
```
repositories/getting-started.md → Basics
repositories/settings.md → Configure
repositories/pull-requests.md → Collaborate
repositories/webhooks.md → Automate
```

### Enterprise & Admin
```
enterprise/overview-and-security.md → Enterprise features
enterprise/audit-logs.md → Track activity
enterprise/billing.md → Manage payments
enterprise/pro-plan.md → PRO features
enterprise/rate-limits.md → API quotas
```

### Advanced Topics
```
advanced/api-endpoints.md → Programmatic access
advanced/security.md → Tokens, GPG, access control
advanced/organizations.md → Team accounts
advanced/agents.md → AI agents on Hub
```

## Common User Questions → Files

| User Question | Read This File |
|---------------|---------------|
| "Upload my trained model" | `models/uploading.md` |
| "Use a model in my code" | `models/downloading.md` + library docs |
| "Run inference without download" | `models/inference.md` |
| "Add interactive demo to model" | `models/widgets.md` or `spaces/overview.md` |
| "Upload a dataset" | `datasets/uploading.md` |
| "Query dataset with SQL" | `datasets/integrations/duckdb.md` |
| "View dataset in browser" | `datasets/data-studio.md` |
| "Create Gradio app" | `spaces/gradio.md` |
| "Deploy Docker container" | `spaces/docker.md` |
| "Add GPU to Space" | `spaces/gpu-upgrades.md` |
| "Control who sees my model" | `models/gated.md` |
| "Control who sees my dataset" | `datasets/gated.md` |
| "Use Hub API" | `advanced/api-endpoints.md` |
| "Set up organization" | `advanced/organizations.md` |
| "Enterprise SSO" | `enterprise/overview-and-security.md` |
| "Check API rate limits" | `enterprise/rate-limits.md` |
| "Track organization activity" | `enterprise/audit-logs.md` |

## Tips for Using This Skill

1. **Start with INDEX.md** if unsure which file to read
2. **Read full files** when possible (they're well-structured)
3. **Combine files** for complex workflows (e.g., upload + inference)
4. **Check library integrations** for framework-specific guidance
5. **Reference code examples** directly from docs when helping users

## Skill Limitations

This skill contains:
- ✅ Hub platform documentation (models, datasets, spaces, repos)
- ✅ Hub API and integration guides
- ✅ Enterprise features documentation

This skill does NOT contain:
- ❌ Transformers library API reference (use transformers docs)
- ❌ Datasets library API reference (use datasets docs)
- ❌ Diffusers library API reference (use diffusers docs)
- ❌ Training/fine-tuning tutorials (use library-specific docs)

For library-specific details, direct users to:
- Transformers: https://huggingface.co/docs/transformers
- Datasets: https://huggingface.co/docs/datasets
- Diffusers: https://huggingface.co/docs/diffusers

## Keeping Docs Updated

These docs were extracted from the official HuggingFace Hub documentation at https://huggingface.co/docs/hub/.

If documentation seems outdated, suggest:
1. Using WebFetch to check current online docs
2. Using Bright Data MCP to scrape latest docs
3. Noting any discrepancies for the user

## Example Usage

**User**: "I want to upload my fine-tuned BERT model to HuggingFace"

**Claude**:
1. Reads `models/uploading.md`
2. Reads `models/model-cards.md`
3. Provides step-by-step guide with code examples
4. Suggests adding model card for documentation

**User**: "How do I query a dataset with SQL without downloading it?"

**Claude**:
1. Reads `datasets/integrations/duckdb.md`
2. Explains DuckDB integration
3. Provides SQL query examples
4. Notes authentication requirements if private dataset

## File Organization

All documentation files are organized by category and use clear, descriptive names. The structure mirrors the official HuggingFace Hub docs structure for easy reference.

See [INDEX.md](INDEX.md) for complete file tree and navigation.
