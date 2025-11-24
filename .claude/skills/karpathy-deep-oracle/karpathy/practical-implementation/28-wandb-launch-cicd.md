# W&B Launch CI/CD Integration

Automate your ML training pipelines with W&B Launch integrated into CI/CD workflows using GitHub Actions, GitLab CI, and webhook-based automation.

## Overview

W&B Launch enables seamless integration with CI/CD systems to create fully automated ML pipelines. By combining Launch with GitHub Actions, GitLab CI, and W&B Automations, you can trigger training jobs, evaluate models, and deploy to production automatically based on code changes, data updates, or model performance thresholds.

**Key capabilities:**
- Automated training on code commits
- Webhook-triggered model evaluations
- GitOps-based deployment workflows
- Integration with GitHub Actions and GitLab CI
- Event-driven pipeline automation

From [W&B Launch Documentation](https://docs.wandb.ai/guides/launch/walkthrough/) (accessed 2025-01-31):
> W&B Launch runs machine learning workloads in containers. Launch is designed to help teams build workflows around shared compute.

---

## Section 1: CI/CD Integration Fundamentals

### GitHub Actions + W&B Launch

**Basic workflow structure:**

```yaml
# .github/workflows/train-model.yml
name: Train Model with W&B Launch

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'data/**'

  workflow_dispatch:
    inputs:
      config_override:
        description: 'Training config override'
        required: false

env:
  WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  WANDB_ENTITY: your-team
  WANDB_PROJECT: production-training

jobs:
  launch-training:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install W&B
        run: pip install wandb>=0.17.1

      - name: Submit training job to Launch queue
        run: |
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --git-hash ${{ github.sha }} \
            --queue gpu-training \
            --project $WANDB_PROJECT \
            --entity $WANDB_ENTITY \
            --config training_config.yaml

      - name: Report status
        if: always()
        run: |
          echo "Training job submitted to W&B Launch queue"
          echo "Monitor at: https://wandb.ai/$WANDB_ENTITY/launch"
```

**Key features:**
- Trigger on push to main branch or manual dispatch
- Authenticate with W&B using secrets
- Submit jobs to Launch queue with specific git commit
- Track in W&B Launch UI

From [CI/CD for Machine Learning Course](https://wandb.ai/site/courses/cicd/) (accessed 2025-01-31):
> Streamline your ML workflows and save valuable time by automating your pipelines and deploying models with confidence. Learn how to use GitHub Actions and integrate W&B experiment tracking.

### GitLab CI Integration

**GitLab CI pipeline:**

```yaml
# .gitlab-ci.yml
stages:
  - test
  - train
  - evaluate
  - deploy

variables:
  WANDB_ENTITY: your-team
  WANDB_PROJECT: production-training

test-code:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - pytest tests/

train-model:
  stage: train
  image: python:3.10
  script:
    - pip install wandb>=0.17.1
    - wandb launch
        --uri https://gitlab.com/$CI_PROJECT_PATH
        --git-hash $CI_COMMIT_SHA
        --queue gpu-training
        --project $WANDB_PROJECT
  only:
    - main
  when: manual  # Require manual approval for training

evaluate-model:
  stage: evaluate
  image: python:3.10
  script:
    - pip install wandb>=0.17.1
    - wandb launch
        --uri https://gitlab.com/$CI_PROJECT_PATH
        --git-hash $CI_COMMIT_SHA
        --queue evaluation
        --entry-point eval.py
  needs:
    - train-model
```

**Trigger patterns:**
- **On push**: Automatic testing and training
- **On PR/MR**: Validation before merge
- **Scheduled**: Periodic retraining (cron jobs)
- **Manual**: Gated training with approval

### Secret Management

**GitHub Actions secrets:**

```yaml
# Set in GitHub repo settings > Secrets
secrets:
  WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  HUGGINGFACE_TOKEN: ${{ secrets.HF_TOKEN }}
  AWS_ACCESS_KEY: ${{ secrets.AWS_KEY }}
```

**GitLab CI variables:**

```yaml
# Set in GitLab repo settings > CI/CD > Variables
variables:
  WANDB_API_KEY: $WANDB_API_KEY  # Masked and protected
  AWS_CREDENTIALS: $AWS_CREDS    # File type variable
```

**Launch job secrets:**

Jobs receive secrets via environment variables configured in Launch queue settings or job config:

```yaml
# launch_config.yaml
resource: kubernetes
queue: gpu-training
environment:
  WANDB_API_KEY: ${WANDB_API_KEY}
  HF_TOKEN: ${HF_TOKEN}
secrets:
  - AWS_CREDENTIALS
  - DATABASE_URL
```

From [W&B Launch FAQ](https://docs.wandb.ai/platform/launch/launch-faq/secrets_jobsautomations_instance_api_key_wish_directly_visible) (accessed 2025-01-31):
> You can specify secrets for jobs/automations. For instance, an API key which you do not wish to be directly visible to users can be managed through environment variables and Launch configurations.

---

## Section 2: Automated Training Pipelines

### Code Change → Automatic Retrain

**Complete GitHub Actions workflow:**

```yaml
# .github/workflows/auto-retrain.yml
name: Automatic Model Retraining

on:
  push:
    branches: [main]
    paths:
      - 'src/**/*.py'
      - 'configs/training_config.yaml'

jobs:
  smoke-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest

      - name: Run smoke tests
        run: pytest tests/smoke/ -v

      - name: Validate training config
        run: python scripts/validate_config.py configs/training_config.yaml

  launch-training:
    needs: smoke-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install W&B
        run: pip install wandb>=0.17.1

      - name: Submit training job
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          # Create job from git URI with current commit
          wandb launch \
            --uri . \
            --git-hash ${{ github.sha }} \
            --queue gpu-training-a100 \
            --project llm-training \
            --entity ${{ vars.WANDB_ENTITY }} \
            --config configs/training_config.yaml \
            --build-context . \
            --dockerfile Dockerfile.training

      - name: Comment on commit
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.repos.createCommitComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              commit_sha: context.sha,
              body: '🚀 Training job submitted to W&B Launch queue: gpu-training-a100\n\nMonitor at: https://wandb.ai/${{ vars.WANDB_ENTITY }}/launch'
            })

  monitor-training:
    needs: launch-training
    runs-on: ubuntu-latest
    steps:
      - name: Monitor training progress
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          # Script to monitor training run and report status
          python scripts/monitor_training.py \
            --entity ${{ vars.WANDB_ENTITY }} \
            --project llm-training \
            --git-commit ${{ github.sha }}
```

**Data drift detection trigger:**

```yaml
# .github/workflows/data-drift-retrain.yml
name: Data Drift Triggered Retraining

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

  repository_dispatch:
    types: [data-drift-detected]

jobs:
  check-data-drift:
    runs-on: ubuntu-latest
    outputs:
      drift_detected: ${{ steps.drift.outputs.detected }}
    steps:
      - uses: actions/checkout@v3

      - name: Check for data drift
        id: drift
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          # Check data distribution metrics in W&B
          python scripts/check_data_drift.py \
            --threshold 0.15 \
            --output detected.txt

          DETECTED=$(cat detected.txt)
          echo "detected=$DETECTED" >> $GITHUB_OUTPUT

  retrain-if-drift:
    needs: check-data-drift
    if: needs.check-data-drift.outputs.drift_detected == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Launch retraining job
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          wandb launch \
            --uri . \
            --queue gpu-training \
            --config configs/retrain_config.yaml \
            --note "Triggered by data drift detection at $(date)"
```

### Checkpoint Validation

**Pre-training validation:**

```yaml
# .github/workflows/validate-before-train.yml
jobs:
  validate-checkpoint:
    runs-on: ubuntu-latest
    steps:
      - name: Download and validate checkpoint
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python scripts/validate_checkpoint.py \
            --checkpoint ${{ inputs.checkpoint_artifact }} \
            --checks integrity,compatibility,performance

      - name: Launch training only if valid
        if: success()
        run: |
          wandb launch --uri . --queue training
```

### Automated Evaluation After Training

**Post-training evaluation workflow:**

```python
# scripts/post_training_eval.py
import wandb
import argparse

def trigger_evaluation(run_id, entity, project):
    """Trigger evaluation job when training completes"""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Check if training succeeded
    if run.state == "finished" and run.summary.get("final_loss") is not None:
        # Submit evaluation job to Launch queue
        wandb.launch(
            uri="https://github.com/your-org/eval-suite",
            queue="evaluation",
            config={
                "model_artifact": run.summary["best_model_artifact"],
                "eval_dataset": "validation-v2",
                "metrics": ["accuracy", "f1", "perplexity"]
            },
            project=project,
            entity=entity
        )
        print(f"✓ Evaluation job submitted for run {run_id}")
    else:
        print(f"✗ Training run {run_id} did not complete successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    trigger_evaluation(args.run_id, args.entity, args.project)
```

**GitHub Actions integration:**

```yaml
# .github/workflows/eval-after-training.yml
name: Evaluate Trained Model

on:
  workflow_run:
    workflows: ["Train Model with W&B Launch"]
    types: [completed]

jobs:
  trigger-evaluation:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Trigger evaluation
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          # Extract training run ID from previous workflow
          RUN_ID=$(echo '${{ github.event.workflow_run.head_commit.message }}' | grep -oP 'run_id:\s*\K\w+')

          python scripts/post_training_eval.py \
            --run-id $RUN_ID \
            --entity ${{ vars.WANDB_ENTITY }} \
            --project ${{ vars.WANDB_PROJECT }}
```

From [W&B Automations Tutorial](https://wandb.ai/examples/wandb_automations/reports/A-Tutorial-on-Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx) (accessed 2025-01-31):
> You'll learn how to use automations in Weights & Biases to trigger automatic evaluation of new model candidates, so you can easily compare performance with a standardized evaluation suite.

---

## Section 3: Model Deployment Workflows

### Training → Evaluation → Staging → Production

**Complete deployment pipeline:**

```yaml
# .github/workflows/model-deployment-pipeline.yml
name: Model Deployment Pipeline

on:
  workflow_dispatch:
    inputs:
      model_artifact:
        description: 'Model artifact to deploy'
        required: true
      environment:
        description: 'Deployment environment'
        required: true
        type: choice
        options:
          - staging
          - production

jobs:
  evaluate-model:
    runs-on: ubuntu-latest
    outputs:
      passed: ${{ steps.eval.outputs.passed }}
    steps:
      - uses: actions/checkout@v3

      - name: Run evaluation suite
        id: eval
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --entry-point eval.py \
            --queue evaluation \
            --config evaluation_config.yaml \
            --override model_artifact=${{ inputs.model_artifact }} \
            --wait  # Wait for evaluation to complete

          # Check evaluation results
          python scripts/check_eval_results.py \
            --artifact ${{ inputs.model_artifact }} \
            --thresholds thresholds.json \
            --output passed.txt

          PASSED=$(cat passed.txt)
          echo "passed=$PASSED" >> $GITHUB_OUTPUT

  deploy-to-staging:
    needs: evaluate-model
    if: needs.evaluate-model.outputs.passed == 'true' && inputs.environment == 'staging'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --entry-point deploy.py \
            --queue deployment-staging \
            --config deployment_config_staging.yaml \
            --override model_artifact=${{ inputs.model_artifact }}

  regression-tests:
    needs: deploy-to-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run regression tests
        run: |
          pytest tests/regression/ \
            --model-endpoint https://staging.api.yourcompany.com/v1/predict \
            --verbose

  deploy-to-production:
    needs: [evaluate-model, regression-tests]
    if: needs.evaluate-model.outputs.passed == 'true' && inputs.environment == 'production'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --entry-point deploy.py \
            --queue deployment-production \
            --config deployment_config_production.yaml \
            --override model_artifact=${{ inputs.model_artifact }}

      - name: Update model registry
        run: |
          python scripts/promote_model.py \
            --artifact ${{ inputs.model_artifact }} \
            --stage production \
            --changelog "Deployed via GitHub Actions on $(date)"
```

### Automated Model Promotion

**W&B Webhooks + GitHub Actions:**

```yaml
# .github/workflows/webhook-model-promotion.yml
name: Webhook Model Promotion

on:
  repository_dispatch:
    types: [model-linked-to-registry]

jobs:
  auto-promote:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Extract model info
        id: model
        run: |
          echo "artifact=${{ github.event.client_payload.artifact }}" >> $GITHUB_OUTPUT
          echo "alias=${{ github.event.client_payload.alias }}" >> $GITHUB_OUTPUT

      - name: Run model tests
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --entry-point test_model.py \
            --queue model-testing \
            --override model_artifact=${{ steps.model.outputs.artifact }} \
            --wait

      - name: Promote if tests pass
        if: success()
        run: |
          python scripts/promote_model.py \
            --artifact ${{ steps.model.outputs.artifact }} \
            --from-stage candidate \
            --to-stage production
```

**W&B Webhook configuration:**

From [Model CI/CD with Webhook Automations](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-webhook-automations-on-Weights-Biases--Vmlldzo0OTcwNDQw) (accessed 2025-01-31):
> When new models get linked to the W&B Model Registry, a webhook POST request can automatically trigger a GitHub Action workflow to consume the model, run tests and deploy.

**Setup webhook in W&B:**

```python
# scripts/setup_webhook.py
import wandb
import os

def create_model_registry_webhook():
    """Create webhook to trigger GitHub Actions on model registry events"""
    api = wandb.Api()

    # Webhook endpoint: Your GitHub Actions webhook
    webhook_url = "https://api.github.com/repos/your-org/your-repo/dispatches"

    # W&B webhook configuration
    webhook_config = {
        "event_type": "model-linked-to-registry",
        "url": webhook_url,
        "secret": os.getenv("GITHUB_WEBHOOK_SECRET"),
        "events": [
            "model_linked",
            "model_alias_updated"
        ]
    }

    # Create webhook in team settings
    # (done via W&B UI: Team Settings > Webhooks)
    print("Create webhook in W&B UI with:")
    print(f"URL: {webhook_url}")
    print(f"Events: {webhook_config['events']}")
```

### Rollback Mechanisms

**Automated rollback on failure:**

```yaml
# .github/workflows/rollback.yml
name: Automated Rollback

on:
  workflow_run:
    workflows: ["Model Deployment Pipeline"]
    types: [completed]

jobs:
  check-deployment:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Trigger rollback
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          # Get previous stable model from registry
          PREV_MODEL=$(python scripts/get_previous_model.py --stage production)

          # Deploy previous model
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --entry-point deploy.py \
            --queue deployment-production \
            --override model_artifact=$PREV_MODEL \
            --note "ROLLBACK: Deployment failure detected"

      - name: Create incident issue
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🚨 Production Deployment Failed - Rollback Executed',
              body: 'Deployment failed and automatic rollback was triggered.\n\nCheck logs at: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.event.workflow_run.id }}',
              labels: ['incident', 'production', 'rollback']
            })
```

### Blue-Green Deployment with Launch

**Blue-green deployment strategy:**

```python
# scripts/blue_green_deploy.py
import wandb
import argparse
import time

def blue_green_deploy(model_artifact, entity, project):
    """Deploy model using blue-green strategy"""

    # Deploy to "green" environment (inactive)
    green_job = wandb.launch(
        uri="https://github.com/your-org/deployment",
        entry_point="deploy_green.py",
        queue="deployment-green",
        config={
            "model_artifact": model_artifact,
            "environment": "green",
            "health_check_url": "https://green.api.yourcompany.com/health"
        },
        project=project,
        entity=entity,
        wait=True
    )

    print("✓ Green deployment complete")

    # Run smoke tests on green environment
    test_job = wandb.launch(
        uri="https://github.com/your-org/deployment",
        entry_point="smoke_tests.py",
        queue="testing",
        config={
            "target_url": "https://green.api.yourcompany.com",
            "test_suite": "smoke"
        },
        project=project,
        entity=entity,
        wait=True
    )

    if test_job.state == "finished":
        print("✓ Smoke tests passed")

        # Switch traffic from blue to green
        switch_job = wandb.launch(
            uri="https://github.com/your-org/deployment",
            entry_point="switch_traffic.py",
            queue="deployment-production",
            config={
                "from": "blue",
                "to": "green",
                "strategy": "gradual",  # or "immediate"
                "duration_minutes": 15
            },
            project=project,
            entity=entity
        )

        print("✓ Traffic switch initiated")
        return True
    else:
        print("✗ Smoke tests failed - keeping blue active")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    success = blue_green_deploy(args.model_artifact, args.entity, args.project)
    exit(0 if success else 1)
```

**GitHub Actions workflow:**

```yaml
# .github/workflows/blue-green-deploy.yml
name: Blue-Green Deployment

on:
  workflow_dispatch:
    inputs:
      model_artifact:
        description: 'Model artifact to deploy'
        required: true

jobs:
  blue-green-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Execute blue-green deployment
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python scripts/blue_green_deploy.py \
            --model-artifact ${{ inputs.model_artifact }} \
            --entity ${{ vars.WANDB_ENTITY }} \
            --project production-deployment
```

From [W&B Automations Documentation](https://docs.wandb.ai/models/automations) (accessed 2025-01-31):
> Create an automation to trigger workflow steps, such as automated model testing and deployment, based on an event in W&B.

---

## Complete ARR-COC CI/CD Example

**Full production pipeline for ARR-COC training:**

```yaml
# .github/workflows/arr-coc-training-pipeline.yml
name: ARR-COC Training Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'arr_coc/**'
      - 'configs/training/**'

  schedule:
    - cron: '0 2 * * 0'  # Weekly Sunday 2 AM

  workflow_dispatch:
    inputs:
      dataset_version:
        description: 'Dataset artifact version'
        required: false
      run_ablations:
        description: 'Run ablation studies'
        type: boolean
        default: false

jobs:
  validate-code:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest black mypy

      - name: Code quality checks
        run: |
          black --check arr_coc/
          mypy arr_coc/

      - name: Run unit tests
        run: pytest tests/unit/ -v

  train-arr-coc:
    needs: validate-code
    runs-on: ubuntu-latest
    outputs:
      run_id: ${{ steps.launch.outputs.run_id }}
    steps:
      - uses: actions/checkout@v3

      - name: Install W&B
        run: pip install wandb>=0.17.1

      - name: Launch training job
        id: launch
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          # Submit to A100 GPU queue
          RUN_ID=$(wandb launch \
            --uri . \
            --git-hash ${{ github.sha }} \
            --queue gpu-training-a100 \
            --project arr-coc-training \
            --entity northhead \
            --config configs/training/arr_coc_base.yaml \
            --override dataset_version=${{ inputs.dataset_version || 'latest' }} \
            --json | jq -r '.run_id')

          echo "run_id=$RUN_ID" >> $GITHUB_OUTPUT
          echo "Training run ID: $RUN_ID"

  ablation-studies:
    needs: train-arr-coc
    if: inputs.run_ablations == true
    runs-on: ubuntu-latest
    strategy:
      matrix:
        ablation:
          - no_propositional
          - no_perspectival
          - no_participatory
          - uniform_lod
    steps:
      - name: Launch ablation job
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --queue gpu-training-a100 \
            --project arr-coc-ablations \
            --config configs/ablations/${{ matrix.ablation }}.yaml

  evaluate-arr-coc:
    needs: train-arr-coc
    runs-on: ubuntu-latest
    steps:
      - name: Launch VQA evaluation
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --entry-point eval_vqa.py \
            --queue evaluation \
            --project arr-coc-evaluation \
            --config configs/eval/vqa_benchmark.yaml \
            --override trained_run_id=${{ needs.train-arr-coc.outputs.run_id }}

  comparative-evaluation:
    needs: evaluate-arr-coc
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Compare against baselines
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          # Generate comparative report
          python scripts/compare_baselines.py \
            --run-id ${{ needs.train-arr-coc.outputs.run_id }} \
            --baselines ovis2.5,llava1.6,qwen2vl \
            --metrics accuracy,token_efficiency,relevance_score \
            --output-report comparison_report.md

      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: comparison-report
          path: comparison_report.md

  deploy-model:
    needs: [evaluate-arr-coc, comparative-evaluation]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to HuggingFace Space
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          wandb launch \
            --uri https://github.com/${{ github.repository }} \
            --entry-point deploy_hf_space.py \
            --queue deployment \
            --config configs/deployment/hf_space.yaml \
            --override run_id=${{ needs.train-arr-coc.outputs.run_id }}
```

**Monitoring and reporting:**

```python
# scripts/monitor_training.py
import wandb
import argparse
import time

def monitor_training(entity, project, git_commit):
    """Monitor training run and report progress to GitHub"""
    api = wandb.Api()

    # Find run by git commit
    runs = api.runs(f"{entity}/{project}", filters={"config.git_commit": git_commit})

    if not runs:
        print(f"No runs found for commit {git_commit}")
        return

    run = runs[0]
    print(f"Monitoring run: {run.id}")

    while run.state in ["running", "pending"]:
        # Refresh run state
        run = api.run(f"{entity}/{project}/{run.id}")

        # Report metrics
        if run.summary:
            print(f"Loss: {run.summary.get('loss', 'N/A')}")
            print(f"Accuracy: {run.summary.get('accuracy', 'N/A')}")
            print(f"Tokens/sec: {run.summary.get('tokens_per_second', 'N/A')}")

        time.sleep(60)  # Check every minute

    print(f"Training complete. Final state: {run.state}")

    if run.state == "finished":
        print(f"✓ Best model: {run.summary.get('best_model_artifact')}")
        return 0
    else:
        print(f"✗ Training failed: {run.summary.get('error_message', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    exit(monitor_training(args.entity, args.project, args.git_commit))
```

---

## Sources

**W&B Documentation:**
- [W&B Launch Walkthrough](https://docs.wandb.ai/guides/launch/walkthrough/) - Launch basics and fundamentals (accessed 2025-01-31)
- [W&B Automations Overview](https://docs.wandb.ai/models/automations) - Event-driven workflow automation (accessed 2025-01-31)
- [Model CI/CD with Webhook Automations](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-webhook-automations-on-Weights-Biases--Vmlldzo0OTcwNDQw) - Webhook-based model deployment (accessed 2025-01-31)
- [W&B Automations Tutorial](https://wandb.ai/examples/wandb_automations/reports/A-Tutorial-on-Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx) - Automated model evaluation patterns (accessed 2025-01-31)

**Courses:**
- [CI/CD for Machine Learning (GitOps)](https://wandb.ai/site/courses/cicd/) - GitHub Actions and W&B integration course (accessed 2025-01-31)

**Community Resources:**
- GitHub Actions documentation - CI/CD workflow syntax and patterns
- GitLab CI documentation - Pipeline configuration and automation
- W&B community forums - CI/CD integration examples

**Additional References:**
- [W&B Launch FAQ](https://docs.wandb.ai/platform/launch/launch-faq/) - Common questions and solutions (accessed 2025-01-31)
- [Connect events to downstream actions with Automations](https://wandb.ai/site/automations/) - Automation use cases (accessed 2025-01-31)
