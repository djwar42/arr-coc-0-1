# Cloud Build Advanced: Security Scanning and Authorization

**Topic**: Container vulnerability scanning, Binary Authorization, and Secret Manager integration for Cloud Build
**Last Updated**: 2025-02-03
**Scope**: Security best practices for Cloud Build pipelines on GCP

---

## Section 1: Security Overview (~100 lines)

### Introduction to Cloud Build Security

Cloud Build security encompasses three critical pillars:

1. **Vulnerability Scanning** - Automated detection of security issues in container images
2. **Binary Authorization** - Policy-based deployment control ensuring only trusted images run
3. **Secret Management** - Secure handling of sensitive data (API keys, passwords, certificates)

These systems work together to create a comprehensive security posture for CI/CD pipelines.

### Security Integration Flow

```
Build Stage → Scan Stage → Attestation Stage → Deploy Stage
    ↓             ↓              ↓                ↓
 Container    Artifact      Binary Auth      GKE/Cloud Run
   Image      Registry       Policy           Deployment
             Scanning       Validation
```

### Why Security Scanning Matters

From [Google Cloud Blog - December 2024](https://security.googleblog.com/2024/12/):
> "We're continually expanding Artifact Analysis capabilities and in 2025 we'll be integrating Artifact Registry vulnerability findings with Google Security Command Center for enhanced visibility."

**Key Statistics**:
- 70%+ of production container images contain known vulnerabilities
- Average time to remediate: 30-60 days without automation
- Critical CVEs can be exploited within hours of disclosure

### Security Layers

**Layer 1: Build Time**
- Source code scanning
- Dependency vulnerability analysis
- Secret detection in code

**Layer 2: Image Time**
- Container image scanning (Artifact Registry)
- OS package vulnerabilities
- Application dependencies

**Layer 3: Deploy Time**
- Binary Authorization policies
- Attestation verification
- Runtime security policies

**Layer 4: Runtime**
- Continuous validation
- Drift detection
- Runtime monitoring

### Recent Updates (2024-2025)

From [Artifact Registry Release Notes](https://docs.cloud.google.com/artifact-registry/docs/release-notes) (accessed 2025-02-03):

**November 19, 2024**: Artifact Registry now provides option to enable/disable vulnerability scanning on individual repositories (previously all-or-nothing at project level)

**July 23, 2024**: Standard tier container OS vulnerability scanning deprecated, scheduled for shutdown July 31, 2025. Users must migrate to premium scanning tier.

---

## Section 2: Vulnerability Scanning (~150 lines)

### Artifact Registry Automatic Scanning

Artifact Registry provides two scanning modes:

**Automatic Scanning** (Recommended for CI/CD):
```yaml
# Enabled at repository creation
gcloud artifacts repositories create my-repo \
  --repository-format=docker \
  --location=us-central1 \
  --enable-vulnerability-scanning
```

**On-Demand Scanning** (For specific images):
```yaml
# Trigger manual scan
gcloud artifacts docker images scan IMAGE_PATH
```

From [Artifact Analysis Container Scanning Overview](https://docs.cloud.google.com/artifact-analysis/docs/container-scanning-overview) (accessed 2025-02-03):
> "Artifact Analysis provides two ways to scan images: automatic scanning and on-demand scanning. Automatic scanning occurs when images are pushed to Artifact Registry."

### Cloud Build Integration

**Enable Container Scanning API**:
```bash
gcloud services enable containerscanning.googleapis.com
```

**cloudbuild.yaml with vulnerability scanning**:
```yaml
steps:
  # Build image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'us-central1-docker.pkg.dev/${PROJECT_ID}/my-repo/app:${SHORT_SHA}', '.']

  # Push to Artifact Registry (triggers automatic scan)
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'us-central1-docker.pkg.dev/${PROJECT_ID}/my-repo/app:${SHORT_SHA}']

  # Wait for scan results (using gcloud alpha)
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        # Poll for scan completion (max 10 minutes)
        for i in {1..60}; do
          SCAN_STATUS=$(gcloud artifacts docker images describe \
            us-central1-docker.pkg.dev/${PROJECT_ID}/my-repo/app:${SHORT_SHA} \
            --show-package-vulnerability \
            --format='value(package_vulnerability_summary.vulnerability_counts.total)')

          if [[ ! -z "$SCAN_STATUS" ]]; then
            echo "Scan complete. Total vulnerabilities: $SCAN_STATUS"
            break
          fi

          echo "Waiting for scan results... (${i}/60)"
          sleep 10
        done

        # Check for CRITICAL vulnerabilities
        CRITICAL=$(gcloud artifacts docker images describe \
          us-central1-docker.pkg.dev/${PROJECT_ID}/my-repo/app:${SHORT_SHA} \
          --show-package-vulnerability \
          --format='value(package_vulnerability_summary.vulnerability_counts.critical)')

        if [[ "$CRITICAL" -gt 0 ]]; then
          echo "ERROR: Found $CRITICAL CRITICAL vulnerabilities!"
          exit 1
        fi

images:
  - 'us-central1-docker.pkg.dev/${PROJECT_ID}/my-repo/app:${SHORT_SHA}'

timeout: 1800s  # 30 minutes (allow time for scanning)
```

### Vulnerability Severity Levels

**CRITICAL** - Immediate action required:
- Remote code execution vulnerabilities
- Authentication bypasses
- Data exposure risks

**HIGH** - Action required within days:
- Privilege escalation
- Information disclosure
- Denial of service

**MEDIUM** - Action required within weeks:
- Cross-site scripting (XSS)
- Local privilege escalation
- Minor information leaks

**LOW** - Monitor and address in regular updates:
- Minor bugs with security implications
- Deprecated functions

### CVE Remediation Workflow

From [Secure Build & Deploy Codelab](https://codelabs.developers.google.com/secure-build-deploy-cloud-build-ar-gke) (accessed 2025-02-03):
> "The scanning service performs vulnerability scans on images in Artifact Registry and Container Registry, then stores the resulting metadata and makes it available through the API."

**Step 1: Identify Vulnerability**
```bash
gcloud artifacts docker images list-vulnerabilities \
  us-central1-docker.pkg.dev/PROJECT_ID/my-repo/app:TAG \
  --format=json
```

**Step 2: Analyze Impact**
- Check affected package/version
- Review CVE database (NVD, vendor advisories)
- Determine exploitability in your context

**Step 3: Remediate**
```dockerfile
# Before (vulnerable)
FROM python:3.9-slim

# After (patched)
FROM python:3.11-slim  # Updated base image with patches
RUN apt-get update && apt-get upgrade -y  # Update system packages
```

**Step 4: Rebuild and Rescan**
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/my-repo/app:v2
```

**Step 5: Verify Fix**
```bash
gcloud artifacts docker images list-vulnerabilities \
  us-central1-docker.pkg.dev/PROJECT_ID/my-repo/app:v2 \
  --occurrence-filter='vulnerability.effectiveSeverity="CRITICAL"'
```

### Scanning Limitations

**What Scanning Detects**:
- Known CVEs in OS packages (apt, yum, apk)
- Known CVEs in language packages (npm, pip, gem, maven)
- Outdated base images

**What Scanning Misses**:
- Zero-day vulnerabilities (not yet in CVE databases)
- Custom application logic flaws
- Configuration issues (exposed ports, weak passwords)
- Supply chain attacks in proprietary code

**Best Practice**: Combine vulnerability scanning with:
- Static Application Security Testing (SAST)
- Dynamic Application Security Testing (DAST)
- Software Composition Analysis (SCA)
- Manual security reviews

---

## Section 3: Binary Authorization (~100 lines)

### Binary Authorization Overview

From [Binary Authorization Overview](https://docs.cloud.google.com/binary-authorization/docs/overview) (accessed 2025-02-03):
> "Binary Authorization is a deploy-time security control that ensures only trusted container images are deployed on Google Kubernetes Engine (GKE) or Cloud Run."

**How It Works**:
```
1. Build image → Cloud Build
2. Scan image → Artifact Registry
3. Attest image → Binary Authorization (cryptographic signature)
4. Deploy image → GKE/Cloud Run checks policy
5. Verify attestation → Allow or deny deployment
```

### Creating Attestations in Cloud Build

From [Create Binary Authorization Attestation Tutorial](https://docs.cloud.google.com/binary-authorization/docs/cloud-build) (accessed 2025-02-03):

**Step 1: Create Attestor**
```bash
# Create attestor
gcloud container binauthz attestors create prod-attestor \
  --attestation-authority-note=projects/${PROJECT_ID}/notes/prod-note \
  --attestation-authority-note-project=${PROJECT_ID}

# Create KMS key for signing
gcloud kms keyrings create binauthz-keyring --location=global

gcloud kms keys create attestor-key \
  --keyring=binauthz-keyring \
  --location=global \
  --purpose=asymmetric-signing \
  --default-algorithm=rsa-sign-pkcs1-4096-sha512

# Associate key with attestor
gcloud container binauthz attestors public-keys add \
  --attestor=prod-attestor \
  --keyversion-project=${PROJECT_ID} \
  --keyversion-location=global \
  --keyversion-keyring=binauthz-keyring \
  --keyversion-key=attestor-key \
  --keyversion=1
```

**Step 2: Create Attestation in Cloud Build**

From [Medium - Binary Authorization Tutorial](https://agrawalkomal.medium.com/google-cloud-binary-authorization-c7c39b5a3135) (accessed 2025-02-03):
> "Binary Authorization provides software supply-chain security for container-based applications by ensuring only verified images from trusted builders are deployed."

```yaml
steps:
  # Build and push image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_IMAGE_PATH}', '.']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_IMAGE_PATH}']

  # Create attestation
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        # Get image digest (not tag - attestations require digest)
        IMAGE_DIGEST=$(gcloud artifacts docker images describe ${_IMAGE_PATH} \
          --format='get(image_summary.digest)')

        IMAGE_URL="${_IMAGE_PATH}@${IMAGE_DIGEST}"

        # Create attestation
        gcloud beta container binauthz attestations sign-and-create \
          --artifact-url="${IMAGE_URL}" \
          --attestor="projects/${PROJECT_ID}/attestors/prod-attestor" \
          --attestor-project="${PROJECT_ID}" \
          --keyversion-project="${PROJECT_ID}" \
          --keyversion-location="global" \
          --keyversion-keyring="binauthz-keyring" \
          --keyversion-key="attestor-key" \
          --keyversion="1"

substitutions:
  _IMAGE_PATH: 'us-central1-docker.pkg.dev/${PROJECT_ID}/my-repo/app:${SHORT_SHA}'

options:
  machineType: 'N1_HIGHCPU_8'
```

### Binary Authorization Policies

**Require Attestations**:
```yaml
# policy.yaml
admissionWhitelistPatterns:
  - namePattern: gcr.io/google-containers/*  # Allow GKE system images

defaultAdmissionRule:
  requireAttestationsBy:
    - projects/PROJECT_ID/attestors/prod-attestor
  enforcementMode: ENFORCED_BLOCK_AND_AUDIT_LOG

globalPolicyEvaluationMode: ENABLE

clusterAdmissionRules:
  us-central1-a.prod-cluster:
    requireAttestationsBy:
      - projects/PROJECT_ID/attestors/prod-attestor
    enforcementMode: ENFORCED_BLOCK_AND_AUDIT_LOG
```

**Apply Policy**:
```bash
gcloud container binauthz policy import policy.yaml
```

### Continuous Validation

From [Continuous Validation Overview](https://docs.cloud.google.com/binary-authorization/docs/overview-cv) (accessed 2025-02-03):
> "Cloud Build is the only trusted builder that the SLSA check supports. Cloud Build must have generated the attestations in either attestation-project1 or attestation-project2."

**What Continuous Validation Checks**:
- Attestations remain valid (not revoked)
- Images haven't been modified post-deployment
- Policies haven't changed to invalidate deployment
- New vulnerabilities discovered post-deployment

---

## Section 4: Secret Management (~100 lines)

### Secret Manager Integration

From [How to Use Secret Manager in Cloud Build](https://ho3einmolavi.medium.com/how-to-use-secret-manager-in-google-cloud-build-gcp-eb6fad9a2d4a) (accessed 2025-02-03):
> "Secret Manager is a secure and convenient storage system for API keys, passwords, certificates, and other sensitive data. Secret Manager provides a central place and single source of truth to manage, access, and audit secrets across Google Cloud."

**Create Secrets**:
```bash
# Create secret
echo -n "my-database-password" | gcloud secrets create db-password \
  --data-file=- \
  --replication-policy="automatic"

# Create API key secret
echo -n "sk-1234567890abcdef" | gcloud secrets create api-key \
  --data-file=-

# Grant Cloud Build access
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='get(projectNumber)')

gcloud secrets add-iam-policy-binding db-password \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding api-key \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Using Secrets in Cloud Build

**Method 1: Environment Variables** (Recommended):
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        # Secrets are available as environment variables
        echo "Building with DB: $$DB_PASSWORD"
        docker build \
          --build-arg DB_PASSWORD=$$DB_PASSWORD \
          --build-arg API_KEY=$$API_KEY \
          -t my-app .
    secretEnv:
      - 'DB_PASSWORD'
      - 'API_KEY'

availableSecrets:
  secretManager:
    - versionName: projects/${PROJECT_ID}/secrets/db-password/versions/latest
      env: 'DB_PASSWORD'
    - versionName: projects/${PROJECT_ID}/secrets/api-key/versions/latest
      env: 'API_KEY'
```

**Method 2: Volume Mounts** (For file-based secrets):
```yaml
steps:
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        # Secret mounted as file
        cat /secrets/service-account-key.json
        gcloud auth activate-service-account --key-file=/secrets/service-account-key.json
    volumes:
      - name: 'secrets'
        path: '/secrets'

availableSecrets:
  secretManager:
    - versionName: projects/${PROJECT_ID}/secrets/sa-key/versions/latest
      env: 'SA_KEY_JSON'
```

**CRITICAL: Secret Access Rules**:

From [Use Secrets Documentation](https://docs.cloud.google.com/build/docs/securing-builds/use-secrets) (accessed 2025-02-03):
> "Add an entrypoint field pointing to bash to use the bash tool in the build step. This is required to refer to the environment variable for the secret. When specifying the secret in the args field, specify it using the environment variable prefixed with $$."

**Why Double $$?**:
- Single `$` → Cloud Build substitution variable
- Double `$$` → Environment variable in build step

**Incorrect** (Won't work):
```yaml
args: ['--password', '$DB_PASSWORD']  # Cloud Build looks for substitution
```

**Correct**:
```yaml
args: ['--password', '$$DB_PASSWORD']  # Bash environment variable
```

### Secret Rotation

**Best Practice**: Rotate secrets regularly:
```bash
# Create new version
echo -n "new-password-value" | gcloud secrets versions add db-password \
  --data-file=-

# Update Cloud Build to use specific version (or use "latest")
# In cloudbuild.yaml:
# versionName: projects/${PROJECT_ID}/secrets/db-password/versions/2

# Disable old version after validation
gcloud secrets versions disable 1 --secret="db-password"

# Destroy old version (irreversible)
gcloud secrets versions destroy 1 --secret="db-password"
```

### Runtime Secret Injection

**For Cloud Run deployments**:
```yaml
steps:
  # Build and push (as before)
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_IMAGE}', '.']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_IMAGE}']

  # Deploy to Cloud Run with secrets
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud run deploy my-service \
          --image=${_IMAGE} \
          --region=us-central1 \
          --set-env-vars=_PORT=$$PORT \
          --set-secrets=DB_PASSWORD=db-password:latest \
          --set-secrets=API_KEY=api-key:latest \
          --allow-unauthenticated
    secretEnv:
      - 'PORT'

availableSecrets:
  secretManager:
    - versionName: projects/${PROJECT_ID}/secrets/port/versions/latest
      env: 'PORT'
```

---

## Section 5: IAM Best Practices (~50 lines)

### Principle of Least Privilege

From [IAM Roles and Permissions](https://docs.cloud.google.com/build/docs/iam-roles-permissions) (accessed 2025-02-03):
> "Permissions are granted by setting policies that grant roles to a principal (user, group, or service account). You can grant multiple roles to a principal on a resource."

**Cloud Build Service Account Permissions**:

**Minimum Required Roles**:
- `roles/storage.objectViewer` - Read source code from GCS
- `roles/logging.logWriter` - Write build logs
- `roles/artifactregistry.writer` - Push images to Artifact Registry

**Additional Roles (as needed)**:
- `roles/secretmanager.secretAccessor` - Access Secret Manager
- `roles/container.developer` - Deploy to GKE
- `roles/run.admin` - Deploy to Cloud Run
- `roles/cloudkms.signerVerifier` - Sign Binary Authorization attestations

**Grant Permissions**:
```bash
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='get(projectNumber)')
SA_EMAIL="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

# Storage access
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectViewer"

# Artifact Registry
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.writer"

# Secret Manager (per-secret basis is more secure)
gcloud secrets add-iam-policy-binding my-secret \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

### Security Best Practices Checklist

**Build Security**:
- [ ] Enable vulnerability scanning on all repositories
- [ ] Block deployments with CRITICAL vulnerabilities
- [ ] Use specific image tags, not `latest`
- [ ] Implement Binary Authorization for production
- [ ] Use Secret Manager for all sensitive data
- [ ] Rotate secrets regularly (30-90 days)
- [ ] Audit Secret Manager access logs

**IAM Security**:
- [ ] Use dedicated service account per environment (dev/staging/prod)
- [ ] Grant minimum required permissions only
- [ ] Use conditions on IAM bindings (time-based, IP-based)
- [ ] Enable Cloud Audit Logs for IAM changes
- [ ] Review permissions quarterly

**Pipeline Security**:
- [ ] Validate build triggers (only trusted branches)
- [ ] Use substitution validation
- [ ] Implement build approval workflows for production
- [ ] Log all build activities
- [ ] Monitor for anomalous build patterns

---

## Sources

**Google Cloud Documentation**:
- [Artifact Analysis Container Scanning](https://docs.cloud.google.com/artifact-analysis/docs/container-scanning-overview) - Container vulnerability scanning overview (accessed 2025-02-03)
- [Binary Authorization Overview](https://docs.cloud.google.com/binary-authorization/docs/overview) - Binary Authorization service documentation (accessed 2025-02-03)
- [Create Binary Authorization Attestation](https://docs.cloud.google.com/binary-authorization/docs/cloud-build) - Tutorial on creating attestations in Cloud Build (accessed 2025-02-03)
- [Use Secrets from Secret Manager](https://docs.cloud.google.com/build/docs/securing-builds/use-secrets) - Secret Manager integration guide (accessed 2025-02-03)
- [IAM Roles and Permissions](https://docs.cloud.google.com/build/docs/iam-roles-permissions) - Cloud Build IAM documentation (accessed 2025-02-03)

**Web Research**:
- [How to Use Secret Manager in Cloud Build](https://ho3einmolavi.medium.com/how-to-use-secret-manager-in-google-cloud-build-gcp-eb6fad9a2d4a) - Medium article by Hossein Molavi (accessed 2025-02-03)
- [Binary Authorization Google Cloud](https://agrawalkomal.medium.com/google-cloud-binary-authorization-c7c39b5a3135) - Medium article by Komal Agrawal (accessed 2025-02-03)
- [Artifact Registry Release Notes](https://docs.cloud.google.com/artifact-registry/docs/release-notes) - Recent updates including November 2024 per-repository scanning control (accessed 2025-02-03)
- [Google Cloud Security Blog - December 2024](https://security.googleblog.com/2024/12/) - 2025 roadmap for Artifact Analysis integration (accessed 2025-02-03)
- [Secure Build & Deploy Codelab](https://codelabs.developers.google.com/secure-build-deploy-cloud-build-ar-gke) - Hands-on tutorial (accessed 2025-02-03)

**Additional References**:
- [Container Vulnerability Scanning Guide](https://www.sentinelone.com/cybersecurity-101/cloud-security/container-vulnerability-scanning/) - SentinelOne overview (accessed 2025-02-03)
- [Use IAM Securely](https://docs.cloud.google.com/iam/docs/using-iam-securely) - GCP IAM best practices (accessed 2025-02-03)
- [Cloud Security Best Practices](https://cloud.google.com/security/best-practices) - Comprehensive security guide (accessed 2025-02-03)
