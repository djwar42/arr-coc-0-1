# Cloud Build Triggers and Automation

## Overview

Cloud Build triggers automate builds in response to various events - repository changes, webhook events, Pub/Sub messages, and scheduled intervals. This document covers trigger types, GitHub/GitLab integration, Pub/Sub event-driven builds, webhook patterns, and workarounds for manual approval workflows (a feature Cloud Build notably lacks).

**Key Concepts:**
- **Source Repository Triggers**: GitHub, GitLab, Bitbucket, Cloud Source Repositories
- **Event-Driven Triggers**: Pub/Sub messages, webhook events
- **Automation Patterns**: Branch filtering, substitution variables, trigger chaining
- **Manual Approval Workarounds**: Two-trigger patterns with Slack integration

---

## Section 1: Trigger Types Overview (~100 lines)

### Available Trigger Types

Cloud Build supports five primary trigger types:

1. **Push to Branch** - Triggers on commits to specific branches
2. **Pull Request** - Triggers on PR creation/updates (GitHub/GitLab)
3. **Tag Push** - Triggers when tags are pushed
4. **Webhook Event** - Triggers via HTTP webhook calls
5. **Pub/Sub Event** - Triggers from Pub/Sub topic messages

### Trigger Configuration Elements

All triggers share common configuration options:

```yaml
# Common trigger configuration
name: trigger-name
description: Human-readable description
disabled: false  # Enable/disable trigger
tags: [production, backend]

# Source configuration
github:
  owner: org-name
  name: repo-name
  push:
    branch: ^main$  # Regex pattern

# Build configuration
filename: cloudbuild.yaml  # Path to build config
substitutions:
  _ENV: production
  _REGION: us-central1

# Service account (optional)
serviceAccount: projects/PROJECT_ID/serviceAccounts/SA_EMAIL

# Advanced options
includedFiles: ['src/**']  # Only trigger if these files change
ignoredFiles: ['.github/**', 'docs/**']  # Ignore these files
```

### Trigger Lifecycle

1. **Event Detection** - Cloud Build detects triggering event
2. **Filter Evaluation** - Check branch patterns, file filters
3. **Build Submission** - Create build with substitutions
4. **Execution** - Run steps in cloudbuild.yaml
5. **Status Update** - Publish to cloud-builds Pub/Sub topic

### Best Practices

**Branch Filtering:**
- Use regex patterns: `^main$` (exact match), `^feature/.*` (prefix match)
- Combine with file filters for monorepo patterns
- Test patterns with Cloud Build's regex tester

**Substitution Variables:**
- Prefix custom variables with `_` (e.g., `_REGION`)
- Use `$(body.payload.field)` for webhook/Pub/Sub data
- Set `substitution_option: ALLOW_LOOSE` for optional variables

**Security:**
- Use service accounts with minimal IAM permissions
- Store secrets in Secret Manager, not substitutions
- Validate webhook signatures (see webhook section)

**Performance:**
- Use `includedFiles`/`ignoredFiles` to avoid unnecessary builds
- Cache dependencies with Kaniko or Docker layer caching
- Use worker pools for network isolation and faster builds

From [Cloud Build Triggers Documentation](https://docs.cloud.google.com/build/docs/triggers) (accessed 2025-02-03)

---

## Section 2: GitHub Integration (~150 lines)

### Connecting GitHub Repositories

Cloud Build supports two GitHub integration methods:

**1. GitHub App (Recommended):**
- OAuth-based, no personal access tokens required
- Fine-grained repository access
- Automatic webhook management
- Works with GitHub Enterprise

**2. GitHub Cloud Build App:**
- Legacy integration method
- Requires admin access to repository
- Manual webhook configuration

### Setup Steps

**Initial Connection:**

```bash
# Navigate to Cloud Build triggers
gcloud builds triggers create github \
  --name="github-push-trigger" \
  --repo-owner="org-name" \
  --repo-name="repo-name" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml"

# Or use console UI:
# 1. Cloud Build → Triggers → Connect Repository
# 2. Select GitHub → Authorize Cloud Build
# 3. Choose repositories to connect
```

**Trigger on Push:**

```yaml
# Example: Deploy on main branch push
name: github-deploy-main
github:
  owner: my-org
  name: my-repo
  push:
    branch: ^main$

filename: deploy/cloudbuild.yaml

substitutions:
  _DEPLOY_ENV: production
  _REGION: us-central1
  _SHORT_SHA: $(short_sha)  # Built-in variable

includedFiles:
  - 'src/**'
  - 'deploy/**'

ignoredFiles:
  - 'docs/**'
  - 'README.md'
```

**Trigger on Pull Request:**

```yaml
# Example: Run tests on PR
name: github-pr-tests
github:
  owner: my-org
  name: my-repo
  pullRequest:
    branch: ^main$  # Target branch regex
    commentControl: COMMENTS_ENABLED  # Require /gcbrun comment
    invertRegex: false

filename: test/cloudbuild.yaml

# Comment control options:
# - COMMENTS_DISABLED: Auto-run on every PR update
# - COMMENTS_ENABLED: Require /gcbrun comment
# - COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY
```

**Trigger on Tag:**

```yaml
# Example: Release on version tag
name: github-release
github:
  owner: my-org
  name: my-repo
  push:
    tag: ^v[0-9]+\.[0-9]+\.[0-9]+$  # Match v1.2.3 format

filename: release/cloudbuild.yaml

substitutions:
  _TAG_NAME: $(tag_name)
```

### Advanced GitHub Patterns

**Monorepo with Multiple Services:**

```yaml
# Trigger service-a builds only when service-a code changes
name: github-service-a
github:
  owner: my-org
  name: monorepo
  push:
    branch: ^main$

filename: services/service-a/cloudbuild.yaml

includedFiles:
  - 'services/service-a/**'
  - 'shared/common/**'  # Shared dependencies

ignoredFiles:
  - '**/*.md'
  - '**/tests/**'
```

**Multi-Stage Deployment Pipeline:**

```yaml
# Stage 1: Build and push image (on PR merge)
name: github-build-stage
github:
  owner: my-org
  name: my-repo
  push:
    branch: ^main$

filename: build/cloudbuild.yaml

# Stage 2: Deploy to staging (separate trigger)
name: github-deploy-staging
# ... configure as needed
```

**Security: Check Run Annotations:**

Cloud Build can post check run results to GitHub PRs:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'my-image', '.']

  # Security scanning with annotations
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'beta'
      - 'builds'
      - 'submit'
      - '--no-source'
      - '--config=security-scan.yaml'

# Results appear as GitHub Check Run annotations
```

### Troubleshooting GitHub Integration

**Common Issues:**

1. **"Repository not found"**
   - Verify Cloud Build app has access to repository
   - Re-authorize GitHub connection
   - Check organization/team access settings

2. **Triggers not firing:**
   - Verify webhook is active in GitHub repo settings
   - Check branch/tag regex patterns
   - Review file inclusion/exclusion filters
   - Check Cloud Build service account IAM permissions

3. **Slow trigger response:**
   - GitHub webhook delivery can take 5-30 seconds
   - Check GitHub webhook delivery logs
   - Consider using worker pools for faster execution

From [Cloud Build GitHub Integration](https://docs.cloud.google.com/build/docs/automating-builds/create-manage-triggers) (accessed 2025-02-03)

---

## Section 3: Pub/Sub Triggers (~100 lines)

### Event-Driven Builds with Pub/Sub

Cloud Build Pub/Sub triggers enable event-driven architectures - respond to Google Cloud events like Cloud Storage changes, Eventarc events, or custom application events.

**Use Cases:**
- Rebuild images when base image updates
- Process uploaded files in Cloud Storage
- React to Cloud Run deployments
- Trigger builds from Cloud Scheduler
- Chain builds across multiple projects

### Creating Pub/Sub Triggers

**Basic Pub/Sub Trigger:**

```bash
# Create topic (if not exists)
gcloud pubsub topics create cloud-build-trigger-topic

# Create trigger
gcloud builds triggers create pubsub \
  --name="pubsub-trigger" \
  --topic="projects/PROJECT_ID/topics/cloud-build-trigger-topic" \
  --build-config="cloudbuild.yaml" \
  --substitutions="_ENV=staging"
```

**Console UI Setup:**
1. Cloud Build → Triggers → Create Trigger
2. Event: Choose "Pub/Sub message"
3. Select existing topic or create new
4. Configure build config and substitutions

### Accessing Pub/Sub Message Data

Pub/Sub messages contain data and attributes accessible via substitutions:

```yaml
# cloudbuild.yaml with Pub/Sub data
steps:
  - name: 'bash'
    args:
      - '-c'
      - |
        echo "Message data: $(body.message.data)"
        echo "Attribute value: $(body.message.attributes.key)"

substitutions:
  # Access base64-encoded message data
  _MESSAGE_DATA: $(body.message.data)

  # Access message attributes
  _CUSTOM_ATTR: $(body.message.attributes.customKey)

  # Access publishing time
  _PUBLISH_TIME: $(body.message.publishTime)

options:
  substitution_option: ALLOW_LOOSE
```

### Publishing Pub/Sub Messages

**From gcloud:**

```bash
# Publish message with data
gcloud pubsub topics publish cloud-build-trigger-topic \
  --message='{"repo": "my-app", "version": "1.2.3"}' \
  --attribute=env=production,region=us-central1
```

**From Cloud Function (Node.js):**

```javascript
const { PubSub } = require('@google-cloud/pubsub');
const pubsub = new PubSub();

async function triggerBuild(buildData) {
  const topic = pubsub.topic('cloud-build-trigger-topic');

  const messageObject = {
    repo: buildData.repo,
    version: buildData.version
  };

  const messageBuffer = Buffer.from(JSON.stringify(messageObject));

  await topic.publishMessage({
    data: messageBuffer,
    attributes: {
      env: buildData.environment,
      triggeredBy: 'cloud-function'
    }
  });
}
```

### Event-Driven Patterns

**Pattern 1: Cloud Storage Upload Trigger**

```bash
# Create Cloud Function that publishes to Pub/Sub on file upload
gcloud functions deploy storage-to-pubsub \
  --runtime=nodejs18 \
  --trigger-bucket=my-upload-bucket \
  --entry-point=publishToPubSub
```

**Pattern 2: Cloud Scheduler for Nightly Builds**

```bash
# Schedule daily build at 2 AM UTC
gcloud scheduler jobs create pubsub nightly-build \
  --schedule="0 2 * * *" \
  --topic=cloud-build-trigger-topic \
  --message-body='{"buildType": "nightly", "fullTest": true}' \
  --attributes=priority=high,env=staging
```

**Pattern 3: Multi-Project Build Orchestration**

```yaml
# Project A: Publish event after successful build
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'image-a', '.']

  # Trigger Project B build
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'pubsub'
      - 'topics'
      - 'publish'
      - 'projects/PROJECT_B/topics/build-trigger'
      - '--message={"upstream":"project-a","image":"image-a:${SHORT_SHA}"}'
```

### Pub/Sub Trigger Security

**IAM Permissions Required:**

```bash
# Grant Cloud Build service account Pub/Sub permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/pubsub.subscriber"

# For publishing (if builds publish messages)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/pubsub.publisher"
```

**Message Validation:**

```yaml
steps:
  # Validate message schema before processing
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        MESSAGE='$(body.message.data)'
        if [ -z "$MESSAGE" ]; then
          echo "Error: Empty message"
          exit 1
        fi
        # Additional validation logic
```

From [Automate Builds in Response to Pub/Sub Events](https://docs.cloud.google.com/build/docs/automate-builds-pubsub-events) (accessed 2025-02-03)

---

## Section 4: Manual Approvals (~100 lines)

### The Manual Approval Problem

**Cloud Build does NOT have native manual approval step functionality.** This is a significant limitation compared to CI/CD tools like GitLab, Jenkins, or AWS CodePipeline.

**Common Use Cases Requiring Manual Approval:**
- Infrastructure changes (review Terraform plan before apply)
- Production deployments
- Database migrations
- Security-sensitive operations

### Workaround: Two-Trigger Pattern with Slack

Since Cloud Build lacks manual approval, the standard workaround is:

1. **Trigger 1 (Plan)**: Run on git push, generate plan output, send to Slack
2. **Human Review**: Reviewers examine plan in Slack channel
3. **Manual Approval**: Reviewer uses Slack slash command to trigger apply
4. **Trigger 2 (Apply)**: Webhook trigger executes the approved plan

**Architecture:**

```
Git Push → Plan Trigger → Generate Plan → Upload to JFrog
                              ↓
                     Send Slack Message
                              ↓
                    Reviewer Examines Plan
                              ↓
              Slack Slash Command → Webhook Trigger
                              ↓
         Apply Trigger → Download Plan → Execute Apply
```

### Implementation: Terraform Plan/Apply Example

**Trigger 1: Terraform Plan (Push to Branch)**

```yaml
# tf-plan.yaml
steps:
  # Initialize Terraform
  - name: 'hashicorp/terraform'
    args: ['init']
    dir: 'terraform/'

  # Format check
  - name: 'hashicorp/terraform'
    args: ['fmt', '-check']
    dir: 'terraform/'

  # Generate plan
  - name: 'hashicorp/terraform'
    args: ['plan', '-out=tfplan']
    dir: 'terraform/'

  # Convert plan to JSON for storage
  - name: 'hashicorp/terraform'
    args: ['show', '-json', 'tfplan']
    dir: 'terraform/'
    env:
      - 'PLAN_JSON=/workspace/tfplan.json'

  # Upload plan to JFrog Artifactory
  - name: 'gcr.io/cloud-builders/curl'
    args:
      - '-X'
      - 'PUT'
      - '-H'
      - 'X-JFrog-Art-Api: ${_JFROG_API_KEY}'
      - '-T'
      - 'terraform/tfplan'
      - '${_JFROG_ARTIFACTORY_URL}/tfplan-${BUILD_ID}'

  # Send Slack notification with approval prompt
  - name: 'gcr.io/cloud-builders/curl'
    args:
      - '-X'
      - 'POST'
      - '-H'
      - 'Content-Type: application/json'
      - '-d'
      - |
        {
          "channel": "${_SLACK_CHANNEL}",
          "text": "Terraform Plan Ready for Review",
          "blocks": [
            {
              "type": "section",
              "text": {
                "type": "mrkdwn",
                "text": "*Terraform Plan Build ${BUILD_ID}*\n\nReview the plan and approve with: `/approve-terraform ${BUILD_ID}`\n\n<https://console.cloud.google.com/cloud-build/builds/${BUILD_ID}|View Build Logs>"
              }
            }
          ]
        }
      - '${_SLACK_WEBHOOK_URL}'

substitutions:
  _JFROG_API_KEY: 'secret-from-secret-manager'
  _JFROG_ARTIFACTORY_URL: 'https://your-org.jfrog.io/artifactory/terraform-plans'
  _SLACK_WEBHOOK_URL: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
  _SLACK_CHANNEL: '#terraform-approvals'

options:
  logging: CLOUD_LOGGING_ONLY
```

**Trigger 2: Terraform Apply (Webhook Event)**

```yaml
# tf-apply.yaml
steps:
  # Send pre-apply notification
  - name: 'gcr.io/cloud-builders/curl'
    args:
      - '-X'
      - 'POST'
      - '-H'
      - 'Content-Type: application/json'
      - '-d'
      - |
        {
          "channel": "${_SLACK_CHANNEL}",
          "text": "🚀 Terraform Apply Started for Build ${_PLAN_BUILD_ID}"
        }
      - '${_SLACK_WEBHOOK_URL}'

  # Download plan from JFrog
  - name: 'gcr.io/cloud-builders/curl'
    args:
      - '-X'
      - 'GET'
      - '-H'
      - 'X-JFrog-Art-Api: ${_JFROG_API_KEY}'
      - '-o'
      - 'terraform/tfplan'
      - '${_JFROG_ARTIFACTORY_URL}/tfplan-${_PLAN_BUILD_ID}'

  # Initialize Terraform
  - name: 'hashicorp/terraform'
    args: ['init']
    dir: 'terraform/'

  # Apply the downloaded plan
  - name: 'hashicorp/terraform'
    args: ['apply', 'tfplan']
    dir: 'terraform/'

  # Clean up - delete plan from JFrog
  - name: 'gcr.io/cloud-builders/curl'
    args:
      - '-X'
      - 'DELETE'
      - '-H'
      - 'X-JFrog-Art-Api: ${_JFROG_API_KEY}'
      - '${_JFROG_ARTIFACTORY_URL}/tfplan-${_PLAN_BUILD_ID}'

  # Send success notification
  - name: 'gcr.io/cloud-builders/curl'
    args:
      - '-X'
      - 'POST'
      - '-H'
      - 'Content-Type: application/json'
      - '-d'
      - |
        {
          "channel": "${_SLACK_CHANNEL}",
          "text": "✅ Terraform Apply Completed Successfully for Build ${_PLAN_BUILD_ID}"
        }
      - '${_SLACK_WEBHOOK_URL}'

substitutions:
  _PLAN_BUILD_ID: $(body.plan_build_id)  # From webhook payload
  _JFROG_API_KEY: 'secret-from-secret-manager'
  _JFROG_ARTIFACTORY_URL: 'https://your-org.jfrog.io/artifactory/terraform-plans'
  _SLACK_WEBHOOK_URL: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
  _SLACK_CHANNEL: '#terraform-approvals'

options:
  substitution_option: ALLOW_LOOSE
  logging: CLOUD_LOGGING_ONLY
```

**Slack Slash Command Setup:**

1. Create Slack App at [api.slack.com/apps](https://api.slack.com/apps)
2. Enable Slash Commands
3. Create command `/approve-terraform`
4. Set Request URL to Cloud Build webhook trigger URL:
   ```
   https://cloudbuild.googleapis.com/v1/projects/PROJECT_ID/triggers/TRIGGER_NAME:webhook?key=API_KEY&secret=SECRET
   ```
5. Configure payload transformation:
   ```json
   {
     "plan_build_id": "$(text)",
     "approved_by": "$(user_name)",
     "approved_at": "$(timestamp)"
   }
   ```

### Alternative: Port.io Integration

For more sophisticated approval workflows, integrate with external platforms:

**Port.io Webhook Pattern:**

```yaml
# Port sends webhook to Cloud Build trigger
# Payload includes approval metadata
steps:
  - name: 'bash'
    args:
      - '-c'
      - |
        echo "Approved by: $(body.trigger.by.user.email)"
        echo "Approval timestamp: $(body.trigger.at)"
        echo "Run ID: $(body.context.runId)"

  # Report status back to Port
  - name: 'gcr.io/cloud-builders/curl'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        curl -X PATCH \
          -H 'Content-Type: application/json' \
          -H "Authorization: Bearer ${_PORT_ACCESS_TOKEN}" \
          -d '{"status":"RUNNING", "message": {"run_status": "Build started"}}' \
          'https://api.getport.io/v1/actions/runs/$(body.context.runId)'

substitutions:
  _PORT_ACCESS_TOKEN: 'secret-from-secret-manager'
```

From [Using Slack to Automate Manual Approvals in Google Cloud Build](https://faun.pub/using-slack-to-automate-manual-approvals-in-google-cloud-build-33de00be6a89) (accessed 2025-02-03)
From [Triggering Cloud Build using Webhooks - Port Documentation](https://docs.port.io/actions-and-automations/setup-backend/webhook/cloudbuild-pipeline/) (accessed 2025-02-03)

---

## Section 5: Advanced Filtering (~50 lines)

### Branch and Tag Filtering

Cloud Build uses regex patterns for branch/tag matching:

**Branch Patterns:**

```yaml
# Exact match
push:
  branch: ^main$

# Match feature branches
push:
  branch: ^feature/.*

# Match release branches (release/1.x, release/2.x)
push:
  branch: ^release/[0-9]+\.x$

# Match multiple patterns (requires multiple triggers)
# Trigger 1: main
# Trigger 2: develop
# Trigger 3: ^release/.*
```

**Tag Patterns:**

```yaml
# Semantic versioning (v1.2.3)
push:
  tag: ^v[0-9]+\.[0-9]+\.[0-9]+$

# Pre-release tags (v1.2.3-beta.1)
push:
  tag: ^v[0-9]+\.[0-9]+\.[0-9]+-.*$

# Release candidate tags (v1.2.3-rc1)
push:
  tag: ^v[0-9]+\.[0-9]+\.[0-9]+-rc[0-9]+$
```

### File Filtering

Use glob patterns to trigger builds only when specific files change:

**includedFiles (trigger IF these change):**

```yaml
includedFiles:
  - 'src/**/*.py'           # All Python files in src/
  - 'requirements.txt'       # Exact file
  - 'Dockerfile'             # Exact file
  - 'cloudbuild.yaml'        # Exact file
  - 'configs/*.json'         # JSON files in configs/
```

**ignoredFiles (DON'T trigger if only these change):**

```yaml
ignoredFiles:
  - 'docs/**'               # All documentation
  - '**/*.md'               # All markdown files
  - 'tests/**'              # All test files
  - '.github/**'            # GitHub config
  - 'scripts/local-dev/**'  # Local dev scripts
```

**Combined Pattern (Monorepo):**

```yaml
# Service A trigger
name: service-a-build
includedFiles:
  - 'services/service-a/**'
  - 'shared/common/**'
ignoredFiles:
  - '**/*.md'
  - '**/tests/**'

# Service B trigger
name: service-b-build
includedFiles:
  - 'services/service-b/**'
  - 'shared/common/**'
ignoredFiles:
  - '**/*.md'
  - '**/tests/**'
```

**Important: File Filtering Behavior:**

- If `includedFiles` is set, trigger fires ONLY if those files change
- If `ignoredFiles` is set, trigger fires UNLESS only those files change
- If both are set, `includedFiles` takes precedence
- Patterns are evaluated against full file paths from repo root

### Testing Filters Locally

```bash
# Test regex pattern matching
echo "feature/new-api" | grep -E '^feature/.*'  # Should match
echo "main" | grep -E '^feature/.*'             # Should NOT match

# Test file glob patterns
# Create test file structure and use:
ls services/service-a/**/*.py  # Test glob expansion
```

---

## Sources

**Google Cloud Documentation:**
- [Cloud Build Triggers](https://docs.cloud.google.com/build/docs/triggers) (accessed 2025-02-03)
- [Create and Manage Build Triggers](https://docs.cloud.google.com/build/docs/automating-builds/create-manage-triggers) (accessed 2025-02-03)
- [Automate Builds in Response to Pub/Sub Events](https://docs.cloud.google.com/build/docs/automate-builds-pubsub-events) (accessed 2025-02-03)
- [Automate Builds in Response to Webhook Events](https://docs.cloud.google.com/build/docs/automate-builds-webhook-events) (accessed 2025-02-03)

**Community Resources:**
- [Using Slack to Automate Manual Approvals in Google Cloud Build](https://faun.pub/using-slack-to-automate-manual-approvals-in-google-cloud-build-33de00be6a89) - Rohan Singh, FAUN.dev, Dec 8, 2022 (accessed 2025-02-03)
- [Triggering Cloud Build using Webhooks - Port Documentation](https://docs.port.io/actions-and-automations/setup-backend/webhook/cloudbuild-pipeline/) (accessed 2025-02-03)
- [Cloud Build Release Notes](https://docs.cloud.google.com/build/docs/release-notes) (accessed 2025-02-03)

**Stack Overflow Discussions:**
- [Is there a Google CloudBuild equivalent for Manual Approval Action in AWS CodePipeline?](https://stackoverflow.com/questions/66113981/) (accessed 2025-02-03)
- [Trigger a Cloud Build to only re-deploy GCP Cloud Functions that were updated](https://stackoverflow.com/questions/78235148/) (accessed 2025-02-03)

**Additional References:**
- [Medium: Integrating Application CI/CD Pipelines and Terraform GitOps with Cloud Build](https://medium.com/google-cloud/integrating-our-application-ci-cd-pipelines-and-terraform-gitops-with-cloud-build-35e8d38b8468) - Dazbo (Darren Lester), 6 months ago (accessed 2025-02-03)
