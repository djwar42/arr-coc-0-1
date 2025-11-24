# GPU Security & Compliance for GCP ML Workloads

## Overview

GPU security and compliance for machine learning workloads requires a multi-layered approach encompassing hardware security, network isolation, data protection, and regulatory compliance. This guide covers security configurations for GPU instances, compliance frameworks (HIPAA, SOC 2, PCI-DSS), workload isolation strategies, and audit capabilities for production ML systems on Google Cloud Platform.

**Core security principles for GPU workloads:**
- Defense in depth: Multiple security layers from hardware to application
- Least privilege: Minimal permissions for GPU access and operations
- Data isolation: Separate GPU workloads by sensitivity and compliance requirements
- Encryption everywhere: At-rest, in-transit, and in-memory protection
- Continuous monitoring: Real-time threat detection and audit logging

From [gcp-vertex/16-vpc-service-controls-private.md](../gcp-vertex/16-vpc-service-controls-private.md) (lines 9-11):
> "VPC Service Controls (VPC-SC) and private networking configurations provide enterprise-grade data exfiltration protection and network isolation for Vertex AI workloads. These security controls are essential for HIPAA, PCI-DSS, and SOC 2 compliance requirements."

---

## 1. Shielded VMs for GPU Instances

### 1.1 Shielded VM Architecture

**Shielded VMs** provide verifiable integrity for GPU instances through hardware-based security features that protect against rootkits, bootkits, and kernel-level malware.

**Core components:**
```yaml
shielded_vm_features:
  secure_boot:
    description: "Ensures only authenticated OS software boots"
    protection: "Prevents unauthorized bootloaders and OS kernels"

  virtual_trusted_platform_module:
    description: "vTPM - Hardware-backed cryptographic operations"
    protection: "Stores boot measurements, encryption keys"

  integrity_monitoring:
    description: "Compares current boot measurements against baseline"
    protection: "Detects tampering, unauthorized changes"
```

From [Google Cloud Shielded VM documentation](https://docs.cloud.google.com/compute/shielded-vm/docs/shielded-vm) (accessed 2025-11-16):
> "Shielded VM offers verifiable integrity of your Compute Engine VM instances, so you can be confident your instances haven't been compromised by boot- or kernel-level malware."

**Creating GPU instance with Shielded VM:**
```bash
# Create A100 GPU instance with Shielded VM
gcloud compute instances create gpu-training-shielded \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu121-debian-11 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    # Shielded VM configuration
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --metadata=enable-oslogin=TRUE

# Verify Shielded VM status
gcloud compute instances get-shielded-identity gpu-training-shielded \
    --zone=us-central1-a
```

### 1.2 Shielded VM Features for GPU Workloads

**Secure Boot:**
```yaml
secure_boot:
  functionality:
    - Verifies bootloader signature using platform firmware
    - Validates kernel signature before execution
    - Blocks unsigned or tampered boot components

  gpu_implications:
    - NVIDIA driver must be signed (official drivers work)
    - Custom kernel modules require signing
    - CUDA toolkit installation validated

  configuration:
    enabled_by_default: false
    recommendation: "Enable for production GPU workloads"
    impact: "Minimal performance overhead (~1-2%)"
```

**vTPM (Virtual Trusted Platform Module):**
```bash
# Enable vTPM for GPU instance
gcloud compute instances create gpu-vtpm-enabled \
    --zone=us-west2-b \
    --machine-type=a2-highgpu-8g \
    --accelerator=type=nvidia-tesla-a100,count=8 \
    --shielded-vtpm \
    --metadata=enable-guest-attributes=TRUE

# Access vTPM measurements (boot integrity)
gcloud compute instances get-shielded-identity gpu-vtpm-enabled \
    --zone=us-west2-b \
    --format=json

# Output includes:
# - encryptionKey: vTPM-backed encryption key
# - kind: Shielded instance identity
# - signingKey: Boot measurement signing key
```

**Integrity Monitoring:**
```python
# Monitor Shielded VM integrity violations
from google.cloud import monitoring_v3
from google.protobuf import duration_pb2

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# Query integrity monitoring violations
interval = monitoring_v3.TimeInterval({
    "end_time": {"seconds": int(time.time())},
    "start_time": {"seconds": int(time.time()) - 86400}  # Last 24 hours
})

results = client.list_time_series(
    request={
        "name": project_name,
        "filter": 'metric.type="compute.googleapis.com/instance/integrity/early_boot_validation_status"',
        "interval": interval,
        "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
    }
)

# Alert on integrity violations
for result in results:
    if result.metric.labels.get("validation_status") == "FAILED":
        print(f"ALERT: Integrity violation on {result.resource.labels['instance_id']}")
        # Trigger incident response, isolate instance
```

From [Massed Compute FAQ](https://massedcompute.com/faq-answers/?question=Can%20I%20use%20Shielded%20VMs%20with%20NVIDIA%20data%20center%20GPUs%20in%20Google%20Cloud?) (accessed 2025-11-16):
> "When creating a VM instance in Google Cloud, you can select a machine type that supports NVIDIA data center GPUs (such as A100, V100, or T4) and enable Shielded VM features. Shielded VM options work with GPU-attached instances."

### 1.3 Shielded VM Best Practices for GPU Training

**Production GPU configuration:**
```bash
# Best practice: Production GPU training instance
gcloud compute instances create prod-gpu-training \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-8g \
    --accelerator=type=nvidia-tesla-a100,count=8 \
    --maintenance-policy=TERMINATE \
    --image-project=deeplearning-platform-release \
    --image-family=common-cu121-debian-11 \
    --boot-disk-size=500GB \
    --boot-disk-type=pd-ssd \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME \
    # Shielded VM (all features enabled)
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    # Security
    --no-address \  # No external IP (use IAP or VPN)
    --network=ml-vpc \
    --subnet=gpu-training-subnet \
    --service-account=gpu-training-sa@project.iam.gserviceaccount.com \
    --scopes=cloud-platform \
    --metadata=enable-oslogin=TRUE,block-project-ssh-keys=TRUE \
    # Labels for governance
    --labels=env=production,workload=training,compliance=hipaa
```

**Modifying Shielded VM options:**
```bash
# Update existing instance (requires stop)
gcloud compute instances stop gpu-training-instance --zone=us-central1-a

# Enable Secure Boot
gcloud compute instances update gpu-training-instance \
    --zone=us-central1-a \
    --shielded-learn-integrity-policy

# Disable integrity monitoring (not recommended for production)
gcloud compute instances update gpu-training-instance \
    --zone=us-central1-a \
    --no-shielded-integrity-monitoring

gcloud compute instances start gpu-training-instance --zone=us-central1-a
```

**arr-coc-0-1 Shielded VM configuration:**
```yaml
# arr-coc-0-1 production GPU instances
arr_coc_shielded_config:
  machine_type: "a2-highgpu-8g"
  gpu: "NVIDIA A100 80GB x8"

  shielded_features:
    secure_boot: enabled
    vtpm: enabled
    integrity_monitoring: enabled

  security_hardening:
    no_external_ip: true
    os_login: true
    block_project_ssh_keys: true

  compliance_labels:
    - "soc2=true"
    - "data-classification=restricted"
    - "gpu-workload=ml-training"
```

---

## 2. Confidential Computing with GPUs

### 2.1 Confidential VMs with H100 GPUs

**Confidential Computing** extends data protection to GPUs, encrypting data in-use during training and inference operations.

From [Google Cloud Confidential Computing](https://cloud.google.com/security/products/confidential-computing) (accessed 2025-11-16):
> "Confidential VMs with H100 GPUs help ensure data remains protected throughout the entire processing pipeline, from the moment it enters the GPU to the final computation results."

**Confidential GPU architecture:**
```yaml
confidential_gpu_features:
  encrypted_memory:
    description: "GPU memory encrypted using AMD SEV or Intel TDX"
    protection: "Data in GPU VRAM protected from physical attacks"

  attestation:
    description: "Cryptographic proof of trusted execution environment"
    protection: "Verify GPU workload runs in secure enclave"

  isolation:
    description: "Hardware-enforced isolation between GPU workloads"
    protection: "Prevent cross-tenant GPU data leakage"
```

**Creating Confidential VM with GPU:**
```bash
# Confidential VM with H100 GPU (Preview)
gcloud compute instances create confidential-gpu-training \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-1g \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --maintenance-policy=TERMINATE \
    --confidential-compute \
    --image-family=ubuntu-2004-lts \
    --image-project=confidential-vm-images \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd

# Verify confidential compute enabled
gcloud compute instances describe confidential-gpu-training \
    --zone=us-central1-a \
    --format="value(confidentialInstanceConfig.enableConfidentialCompute)"
```

From [Google Cloud Confidential VM Release Notes](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/release-notes) (accessed 2025-11-16):
> "July 31, 2025: Support for accelerator-optimized a3-highgpu-1g machine type for securely running AI and ML workloads is now generally available, with support for NVIDIA H100 GPUs in Confidential Computing mode."

### 2.2 GPU Workload Attestation

**Verifying trusted execution:**
```python
# Verify Confidential VM attestation for GPU workload
from google.cloud import compute_v1
import base64
import json

def verify_gpu_attestation(project_id: str, zone: str, instance_name: str):
    """Verify GPU instance runs in confidential mode."""
    client = compute_v1.InstancesClient()

    # Get instance details
    instance = client.get(
        project=project_id,
        zone=zone,
        instance=instance_name
    )

    # Check confidential compute enabled
    if not instance.confidential_instance_config.enable_confidential_compute:
        raise ValueError(f"Instance {instance_name} is not confidential")

    # Get attestation report
    attestation = client.get_shielded_instance_identity(
        project=project_id,
        zone=zone,
        instance=instance_name
    )

    # Verify encryption key
    encryption_key = attestation.encryption_key.ek_pub
    signing_key = attestation.signing_key.ek_pub

    print(f"Attestation verified for {instance_name}")
    print(f"Encryption key: {base64.b64encode(encryption_key).decode()[:50]}...")
    print(f"Signing key: {base64.b64encode(signing_key).decode()[:50]}...")

    return True

# Usage
verify_gpu_attestation("arr-coc-training", "us-central1-a", "confidential-gpu-training")
```

From [Security Google Cloud Community](https://security.googlecloudcommunity.com/community-blog-42/protecting-your-data-why-confidential-computing-is-necessary-for-your-business-4007) (accessed 2025-11-16):
> "We've introduced Confidential GPUs in preview, extending confidential computing to accelerate AI/ML workloads that rely on GPUs while maintaining the highest levels of data protection."

### 2.3 Confidential GPU Limitations and Considerations

**Current limitations (2025):**
```yaml
confidential_gpu_status:
  availability:
    h100_gpus: "Preview (a3-highgpu-1g)"
    a100_gpus: "Not yet supported"
    l4_gpus: "Not yet supported"
    t4_gpus: "Not yet supported"

  regions:
    supported: ["us-central1", "europe-west4"]
    planned: ["us-west1", "asia-southeast1"]

  performance_impact:
    memory_encryption: "5-10% overhead"
    attestation: "Minimal (<1%)"
    overall: "Acceptable for most ML workloads"

  compatibility:
    cuda_versions: "CUDA 12.1+"
    frameworks: "PyTorch 2.0+, TensorFlow 2.13+"
    custom_kernels: "Requires review"
```

**When to use Confidential GPUs:**
```yaml
use_cases:
  recommended:
    - "Healthcare: Training on PHI (Protected Health Information)"
    - "Finance: Fraud detection with PII (Personal Identifiable Information)"
    - "Government: Classified data processing"
    - "Multi-tenant: Isolating customer GPU workloads"

  not_recommended:
    - "Public datasets (no sensitive data)"
    - "Cost-sensitive workloads (5-10% performance penalty)"
    - "Older GPU types (A100, T4 not yet supported)"
```

From [Phala Confidential Computing Trends 2025](https://phala.com/learn/confidential-computing-trends-2025) (accessed 2025-11-16):
> "Gartner predicts 60% of enterprises will evaluate TEE (Trusted Execution Environments) by year-end 2025. GPU TEE adoption accelerating with H100 support on GCP, enabling confidential AI training at scale."

---

## 3. GPU Workload Isolation and Network Security

### 3.1 VPC Service Controls for GPU Workloads

**VPC Service Controls (VPC-SC)** create security perimeters around GPU resources to prevent data exfiltration.

From [gcp-vertex/16-vpc-service-controls-private.md](../gcp-vertex/16-vpc-service-controls-private.md) (lines 14-42):
> "Service Perimeter = Security boundary around Google Cloud resources. Protected Resources include Vertex AI training jobs, GCS buckets (datasets, checkpoints, logs), BigQuery datasets, Artifact Registry repositories."

**GPU training perimeter:**
```yaml
# VPC Service Controls for GPU training
apiVersion: accesscontextmanager.cnrm.cloud.google.com/v1beta1
kind: AccessContextManagerServicePerimeter
metadata:
  name: gpu-training-perimeter
spec:
  title: "GPU ML Training Perimeter"
  perimeterType: PERIMETER_TYPE_REGULAR
  status:
    resources:
      - "projects/gpu-training-project"
    restrictedServices:
      - "aiplatform.googleapis.com"
      - "compute.googleapis.com"
      - "storage.googleapis.com"
      - "artifactregistry.googleapis.com"
    vpcAccessibleServices:
      enableRestriction: true
      allowedServices:
        - "aiplatform.googleapis.com"
        - "compute.googleapis.com"
        - "storage.googleapis.com"
```

**Ingress rules for GPU access:**
```python
# Allow GPU access from trusted corporate network only
ingress_policy = {
    "ingress_from": {
        "sources": [
            {
                "access_level": "accessPolicies/123/accessLevels/corporate_network"
            }
        ],
        "identities": [
            "user:ml-engineer@company.com",
            "serviceAccount:gpu-training-sa@project.iam.gserviceaccount.com"
        ]
    },
    "ingress_to": {
        "resources": ["*"],
        "operations": [
            {
                "service_name": "compute.googleapis.com",
                "method_selectors": [
                    {"method": "compute.instances.create"},  # Create GPU instance
                    {"method": "compute.instances.start"},
                    {"method": "compute.instances.stop"}
                ]
            },
            {
                "service_name": "storage.googleapis.com",
                "method_selectors": [
                    {"method": "google.storage.objects.get"},  # Read training data
                    {"method": "google.storage.objects.create"}  # Write checkpoints
                ]
            }
        ]
    }
}
```

**Egress rules (data exfiltration prevention):**
```python
# Block all egress except approved model registry
egress_policy = {
    "egress_from": {
        "identities": [
            "serviceAccount:gpu-training-sa@project.iam.gserviceaccount.com"
        ]
    },
    "egress_to": {
        "resources": [
            "projects/approved-model-registry"  # ONLY allowed external destination
        ],
        "operations": [
            {
                "service_name": "artifactregistry.googleapis.com",
                "method_selectors": [
                    {"method": "docker.push"}  # Can push trained models
                    # NOTE: docker.pull NOT allowed → cannot download external models
                ]
            }
        ]
    }
}
```

### 3.2 GPU Network Isolation Strategies

**Separate VPCs for GPU workloads:**
```bash
# Create isolated VPC for GPU training
gcloud compute networks create gpu-training-vpc \
    --subnet-mode=custom \
    --bgp-routing-mode=regional

# Create subnet with Private Google Access
gcloud compute networks subnets create gpu-training-subnet \
    --network=gpu-training-vpc \
    --region=us-central1 \
    --range=10.128.0.0/20 \
    --enable-private-ip-google-access \
    --enable-flow-logs

# Create firewall rules (default deny, explicit allow)
# 1. Deny all ingress by default
gcloud compute firewall-rules create deny-all-ingress-gpu \
    --network=gpu-training-vpc \
    --action=DENY \
    --rules=all \
    --direction=INGRESS \
    --priority=65534

# 2. Allow IAP SSH for debugging
gcloud compute firewall-rules create allow-iap-ssh-gpu \
    --network=gpu-training-vpc \
    --action=ALLOW \
    --rules=tcp:22 \
    --source-ranges=35.235.240.0/20 \
    --target-tags=gpu-training \
    --priority=1000

# 3. Allow NCCL for multi-GPU training
gcloud compute firewall-rules create allow-nccl-gpu \
    --network=gpu-training-vpc \
    --action=ALLOW \
    --rules=tcp:1024-65535 \
    --source-tags=gpu-training \
    --target-tags=gpu-training \
    --priority=900
```

**Multi-Instance GPU (MIG) for workload isolation:**
```bash
# Enable MIG on A100 GPU (7 MIG instances)
# SSH into GPU instance
gcloud compute ssh gpu-instance --zone=us-central1-a

# On GPU instance:
# Enable MIG mode
sudo nvidia-smi -mig 1

# Create 7 MIG instances (1g.10gb each)
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C

# Verify MIG instances
nvidia-smi -L
# GPU 0: NVIDIA A100 80GB (UUID: GPU-xxx)
#   MIG 1g.10gb Device 0: (UUID: MIG-xxx)
#   MIG 1g.10gb Device 1: (UUID: MIG-xxx)
#   ...
```

From [Massed Compute FAQ](https://massedcompute.com/faq-answers/?question=What+are+the+best+practices+for+securing+NVIDIA+cloud+services%3F) (accessed 2025-11-16):
> "Use NVIDIA Multi-Instance GPU (MIG) to isolate workloads and prevent cross-tenant interference. Apply firmware updates to GPUs to patch known vulnerabilities."

**MIG isolation benefits:**
```yaml
mig_isolation:
  security:
    - "Hardware-enforced isolation between MIG instances"
    - "Separate memory spaces (no shared VRAM)"
    - "Independent compute resources"

  use_cases:
    - "Multi-tenant GPU sharing (different customers)"
    - "Dev/staging/prod on same GPU (cost savings)"
    - "Isolate sensitive workloads (compliance)"

  limitations:
    - "Only A100 and H100 GPUs support MIG"
    - "Fixed partition sizes (1g, 2g, 3g, 4g, 7g)"
    - "Cannot span MIG instances across GPUs"
```

### 3.3 Private Service Connect for GPU Endpoints

**Eliminating public internet exposure:**
```bash
# Create Private Service Connect endpoint for Vertex AI
gcloud compute addresses create vertex-ai-psc-gpu \
    --region=us-central1 \
    --subnet=gpu-training-subnet \
    --addresses=10.128.0.100

gcloud compute forwarding-rules create vertex-ai-gpu-endpoint \
    --region=us-central1 \
    --network=gpu-training-vpc \
    --address=vertex-ai-psc-gpu \
    --target-service-attachment=projects/cloud-ai-platform-public/regions/us-central1/serviceAttachments/vertex-ai-apis

# Use private endpoint for GPU training
from google.cloud import aiplatform

aiplatform.init(
    project="gpu-training-project",
    location="us-central1",
    api_endpoint="10.128.0.100"  # Private IP, no public internet
)

# GPU training job uses private connectivity
job = aiplatform.CustomJob(
    display_name="private-gpu-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-8g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 8
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-central1-docker.pkg.dev/project/training:latest"
        },
        "network": "projects/gpu-training-project/global/networks/gpu-training-vpc",
        "enable_web_access": False  # No public internet access
    }]
)
```

From [DevZero GPU Security and Isolation](https://www.devzero.io/blog/gpu-security-and-isolation) (accessed 2025-11-16):
> "Effective GPU resource management provides significant security and isolation benefits beyond simple cost optimization. Network isolation, private endpoints, and MIG partitioning are essential for multi-tenant GPU environments."

---

## 4. HIPAA Compliance for GPU Training

### 4.1 HIPAA Requirements for GPU Workloads

**HIPAA (Health Insurance Portability and Accountability Act)** requires specific safeguards for Protected Health Information (PHI) processed on GPUs.

From [gcp-vertex/18-compliance-governance-audit.md](../gcp-vertex/18-compliance-governance-audit.md) (lines 122-150):
> "HIPAA compliance for healthcare: Protected Health Information (PHI) handling requires Business Associate Agreement (BAA) with Google Cloud. Vertex AI HIPAA-eligible services include Custom Jobs, Prediction (Online, Batch), Workbench (Managed Notebooks), and Pipelines."

**HIPAA-compliant GPU architecture:**
```yaml
hipaa_gpu_requirements:
  baa:
    required: true
    scope: "Sign BAA before processing PHI on GPUs"
    services_covered:
      - "Compute Engine (GPU instances)"
      - "Vertex AI Custom Jobs (GPU training)"
      - "Cloud Storage (training data, checkpoints)"
      - "Cloud Logging (audit logs)"

  technical_safeguards:
    encryption:
      at_rest: "CMEK (Customer-Managed Encryption Keys) recommended"
      in_transit: "TLS 1.2+ mandatory"
      in_memory: "Confidential Computing (optional, H100 only)"

    access_controls:
      authentication: "Cloud Identity or Google Workspace"
      mfa: "Mandatory for all PHI access"
      iam: "Least privilege service accounts"

    audit_logging:
      admin_activity: "Enabled by default"
      data_access: "Must enable explicitly for PHI"
      retention: "Minimum 6 years (HIPAA requirement)"
```

**HIPAA GPU training setup:**
```bash
# Step 1: Sign BAA with Google Cloud (contact sales)

# Step 2: Create HIPAA-compliant project
gcloud projects create hipaa-gpu-training \
    --name="HIPAA GPU Training" \
    --labels=compliance=hipaa,data-classification=phi

# Step 3: Enable required services
gcloud services enable compute.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    cloudkms.googleapis.com \
    --project=hipaa-gpu-training

# Step 4: Create CMEK keyring and key
gcloud kms keyrings create hipaa-keyring \
    --location=us-central1 \
    --project=hipaa-gpu-training

gcloud kms keys create phi-encryption-key \
    --location=us-central1 \
    --keyring=hipaa-keyring \
    --purpose=encryption \
    --project=hipaa-gpu-training

# Step 5: Grant GPU service account access to key
gcloud kms keys add-iam-policy-binding phi-encryption-key \
    --location=us-central1 \
    --keyring=hipaa-keyring \
    --member="serviceAccount:gpu-training-sa@hipaa-gpu-training.iam.gserviceaccount.com" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" \
    --project=hipaa-gpu-training

# Step 6: Enable audit logging
cat > audit-config.yaml <<EOF
auditConfigs:
  - service: compute.googleapis.com
    auditLogConfigs:
      - logType: ADMIN_READ
      - logType: DATA_READ
      - logType: DATA_WRITE
  - service: aiplatform.googleapis.com
    auditLogConfigs:
      - logType: ADMIN_READ
      - logType: DATA_READ
      - logType: DATA_WRITE
  - service: storage.googleapis.com
    auditLogConfigs:
      - logType: ADMIN_READ
      - logType: DATA_READ
      - logType: DATA_WRITE
EOF

gcloud projects get-iam-policy hipaa-gpu-training > policy.yaml
# Append audit-config.yaml content to policy.yaml
gcloud projects set-iam-policy hipaa-gpu-training policy.yaml
```

### 4.2 HIPAA GPU Training Job Configuration

**Compliant training job:**
```python
# HIPAA-compliant GPU training on Vertex AI
from google.cloud import aiplatform

aiplatform.init(
    project="hipaa-gpu-training",
    location="us-central1",  # HIPAA-eligible region
    staging_bucket="gs://hipaa-phi-datasets"  # CMEK-encrypted bucket
)

job = aiplatform.CustomTrainingJob(
    display_name="hipaa-compliant-gpu-training",
    container_uri="us-central1-docker.pkg.dev/hipaa-gpu-training/medical-imaging:latest",
    # HIPAA requirements
    requirements=[
        "BAA signed",
        "CMEK encryption enabled",
        "Audit logging enabled",
        "No external IP",
        "VPC Service Controls"
    ]
)

# Run training with HIPAA safeguards
model = job.run(
    dataset="gs://hipaa-phi-datasets/medical-images/train/",
    replica_count=1,
    machine_type="a2-highgpu-8g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=8,
    # CMEK encryption
    training_encryption_spec_key_name="projects/hipaa-gpu-training/locations/us-central1/keyRings/hipaa-keyring/cryptoKeys/phi-encryption-key",
    # Service account (least privilege)
    service_account="gpu-training-sa@hipaa-gpu-training.iam.gserviceaccount.com",
    # Network isolation
    network="projects/hipaa-gpu-training/global/networks/hipaa-vpc",
    enable_web_access=False,  # No public internet
    # Compliance labels
    labels={
        "compliance": "hipaa",
        "data-classification": "phi",
        "environment": "production"
    }
)
```

From [Google Cloud HIPAA Compliance](https://docs.cloud.google.com/security/compliance/hipaa) (accessed 2025-11-16):
> "Google Cloud services covered under HIPAA compliance include Vertex AI, Cloud Storage, BigQuery, and VPC Service Controls when used within a signed Business Associate Agreement (BAA)."

### 4.3 HIPAA Audit and Monitoring

**HIPAA audit requirements:**
```bash
# Query GPU training access logs (HIPAA audit trail)
gcloud logging read '
    resource.type="aiplatform.googleapis.com/CustomJob"
    AND protoPayload.methodName="google.cloud.aiplatform.v1.JobService.CreateCustomJob"
    AND resource.labels.job_id:"hipaa-compliant-gpu-training"
    AND timestamp>="2025-11-01T00:00:00Z"
' \
    --limit=100 \
    --format='table(timestamp, protoPayload.authenticationInfo.principalEmail, protoPayload.request.customJob.displayName, protoPayload.status.code)'

# Export audit logs to BigQuery for 6-year retention (HIPAA requirement)
gcloud logging sinks create hipaa-audit-sink \
    bigquery.googleapis.com/projects/hipaa-gpu-training/datasets/hipaa_audit_logs \
    --log-filter='
        resource.type=("aiplatform.googleapis.com" OR "compute.googleapis.com" OR "storage.googleapis.com")
        AND labels.compliance="hipaa"
    '

# Set BigQuery dataset retention to 6 years
bq update --default_table_expiration 189216000 hipaa-gpu-training:hipaa_audit_logs
# 189216000 seconds = 6 years
```

**HIPAA compliance dashboard:**
```python
# Monitor HIPAA GPU compliance
from google.cloud import monitoring_v3

def create_hipaa_compliance_dashboard(project_id: str):
    """Create Cloud Monitoring dashboard for HIPAA GPU compliance."""
    client = monitoring_v3.DashboardsServiceClient()

    dashboard = monitoring_v3.Dashboard(
        display_name="HIPAA GPU Compliance Dashboard",
        dashboard_filters=[
            monitoring_v3.DashboardFilter(
                label_key="compliance",
                string_value="hipaa"
            )
        ],
        grid_layout=monitoring_v3.GridLayout(
            widgets=[
                # Widget 1: GPU instances without CMEK
                monitoring_v3.GridLayout.Widget(
                    title="GPU Instances Without CMEK Encryption",
                    scorecard=monitoring_v3.Scorecard(
                        time_series_query=monitoring_v3.TimeSeriesQuery(
                            time_series_filter=monitoring_v3.TimeSeriesFilter(
                                filter='resource.type="gce_instance" AND resource.labels.accelerator_type:"nvidia" AND NOT metadata.labels.encryption="cmek"'
                            )
                        ),
                        thresholds=[
                            monitoring_v3.Threshold(value=0.0, color="GREEN"),
                            monitoring_v3.Threshold(value=1.0, color="RED")
                        ]
                    )
                ),

                # Widget 2: Audit log volume
                monitoring_v3.GridLayout.Widget(
                    title="HIPAA Audit Log Events (Last 24h)",
                    xy_chart=monitoring_v3.XyChart(
                        data_sets=[
                            monitoring_v3.XyChart.DataSet(
                                time_series_query=monitoring_v3.TimeSeriesQuery(
                                    time_series_filter=monitoring_v3.TimeSeriesFilter(
                                        filter='resource.type="aiplatform.googleapis.com" AND labels.compliance="hipaa"'
                                    )
                                )
                            )
                        ]
                    )
                ),

                # Widget 3: Instances with external IPs (violation)
                monitoring_v3.GridLayout.Widget(
                    title="GPU Instances with External IPs (VIOLATION)",
                    scorecard=monitoring_v3.Scorecard(
                        time_series_query=monitoring_v3.TimeSeriesQuery(
                            time_series_filter=monitoring_v3.TimeSeriesFilter(
                                filter='resource.type="gce_instance" AND resource.labels.accelerator_type:"nvidia" AND metadata.networkInterfaces.accessConfigs.natIP!=""'
                            )
                        ),
                        thresholds=[
                            monitoring_v3.Threshold(value=0.0, color="GREEN"),
                            monitoring_v3.Threshold(value=1.0, color="RED")
                        ]
                    )
                )
            ]
        )
    )

    # Create dashboard
    project_name = f"projects/{project_id}"
    response = client.create_dashboard(name=project_name, dashboard=dashboard)
    print(f"HIPAA compliance dashboard created: {response.name}")

    return response

# Usage
create_hipaa_compliance_dashboard("hipaa-gpu-training")
```

From [Corvex AI HIPAA GPU Providers](https://www.corvex.ai/blog/top-hipaa-compliant-cloud-gpu-providers-for-secure-ai-model-training) (accessed 2025-11-16):
> "HIPAA-compliant GPU cloud providers must provide: Business Associate Agreement (BAA), encryption at rest and in transit, access controls and audit logging, data isolation and network security, compliance certifications (SOC 2, ISO 27001)."

---

## 5. PCI-DSS and SOC 2 Compliance for GPU Workloads

### 5.1 PCI-DSS Considerations for GPU Training

**PCI-DSS (Payment Card Industry Data Security Standard)** applies when GPU workloads process cardholder data.

From [gcp-vertex/18-compliance-governance-audit.md](../gcp-vertex/18-compliance-governance-audit.md) (lines 222-252):
> "PCI-DSS for payment processing: Cardholder data protection requires Level 1 Service Provider certification. You cannot store Primary Account Numbers (PAN), Card Verification Values (CVV), or unencrypted cardholder data in Vertex AI. Use tokenization before ML processing."

**PCI-DSS GPU architecture:**
```yaml
pci_dss_gpu_requirements:
  data_handling:
    prohibited:
      - "Storing PAN (Primary Account Numbers) on GPU"
      - "Storing CVV (Card Verification Values)"
      - "Training models on raw cardholder data"

    allowed:
      - "Tokenized transaction features"
      - "Hashed merchant IDs"
      - "Anonymized transaction patterns"

  network_segmentation:
    required: "Separate VPC for PCI-scoped GPU workloads"
    firewall: "Default deny, explicit allow only"
    monitoring: "VPC Flow Logs enabled"

  quarterly_scans:
    required: "Vulnerability scanning of GPU instances"
    tool: "Google Cloud Security Command Center"
```

**PCI-DSS compliant fraud detection:**
```python
# Example: Fraud detection WITHOUT cardholder data
import hashlib
from google.cloud import aiplatform

def create_pci_compliant_features(transaction: dict) -> dict:
    """Create ML features without CHD/SAD."""
    return {
        "amount": transaction["amount"],
        "merchant_id_hash": hashlib.sha256(
            transaction["merchant_id"].encode()
        ).hexdigest(),
        "timestamp": transaction["timestamp"],
        "location_hash": hashlib.sha256(
            transaction["location"].encode()
        ).hexdigest(),
        "transaction_type": transaction["type"],
        # NO card numbers, CVV, or cardholder names
    }

# Train fraud detection model on GPU
aiplatform.init(
    project="pci-scoped-project",
    location="us-central1"
)

job = aiplatform.CustomTrainingJob(
    display_name="pci-compliant-fraud-detection",
    container_uri="us-central1-docker.pkg.dev/pci-scoped/fraud-detector:latest"
)

# Model trained on tokenized/hashed features only
model = job.run(
    dataset="gs://tokenized-transactions/train.csv",  # No cardholder data
    replica_count=1,
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    # PCI-DSS network isolation
    network="projects/pci-scoped-project/global/networks/pci-isolated-vpc",
    service_account="pci-gpu-sa@pci-scoped-project.iam.gserviceaccount.com"
)
```

### 5.2 SOC 2 Compliance for GPU Operations

**SOC 2 (Service Organization Control 2)** focuses on security, availability, processing integrity, confidentiality, and privacy.

From [gcp-vertex/18-compliance-governance-audit.md](../gcp-vertex/18-compliance-governance-audit.md) (lines 22-48):
> "Vertex AI SOC 2 compliance covers Training, Prediction, Workbench, Model Registry, Feature Store, Matching Engine, Experiments, and Metadata Store. SOC 2 Type II reports available quarterly, covering operational effectiveness over time."

**SOC 2 compliance checklist for GPUs:**
```yaml
soc2_gpu_compliance:
  security:
    - "Shielded VMs enabled on all GPU instances"
    - "CMEK encryption for boot disks and data"
    - "MFA required for GPU instance access"
    - "IAM least privilege (service accounts)"
    - "VPC Service Controls for data perimeter"

  availability:
    - "Multi-zone GPU deployment (99.9% uptime)"
    - "Automated backups of training checkpoints"
    - "Disaster recovery plan tested quarterly"
    - "SLA monitoring and alerting"

  processing_integrity:
    - "Model versioning in Vertex AI Model Registry"
    - "Training data validation pipelines"
    - "Reproducible training (fixed seeds, versioned code)"
    - "Audit trail of all GPU job submissions"

  confidentiality:
    - "VPC-SC perimeter around GPU resources"
    - "Private Google Access (no external IPs)"
    - "DLP scanning on training datasets"
    - "Access reviews quarterly"

  privacy:
    - "Data minimization (only necessary features)"
    - "Retention policies (auto-delete old checkpoints)"
    - "Right to deletion workflows"
    - "Privacy impact assessments"
```

**SOC 2 GPU monitoring:**
```bash
# Create SOC 2 compliance monitoring
gcloud monitoring policies create \
    --notification-channels=soc2-alerts \
    --display-name="SOC 2 GPU Compliance Violation" \
    --condition-display-name="GPU instance without Shielded VM" \
    --condition-threshold-value=1 \
    --condition-threshold-duration=0s \
    --condition-filter='
        resource.type="gce_instance"
        AND resource.labels.accelerator_type:"nvidia"
        AND NOT metadata.items.key="shielded-integrity-monitoring"
    '

# Alert on unencrypted GPU boot disks
gcloud monitoring policies create \
    --notification-channels=soc2-alerts \
    --display-name="Unencrypted GPU Boot Disk" \
    --condition-display-name="GPU instance without CMEK" \
    --condition-threshold-value=1 \
    --condition-threshold-duration=0s \
    --condition-filter='
        resource.type="gce_instance"
        AND resource.labels.accelerator_type:"nvidia"
        AND NOT metadata.labels.encryption="cmek"
    '
```

From [gcloud-iam/00-service-accounts-ml-security.md](../gcloud-iam/00-service-accounts-ml-security.md) (lines 1-21):
> "IAM is the foundation of secure ML operations on Google Cloud. Service accounts act as both identities for machines and grantees for permissions. For ML workloads on Vertex AI, GKE, or Compute Engine, proper IAM configuration prevents compliance nightmares."

---

## 6. arr-coc-0-1 Security Architecture

### 6.1 Complete Security Configuration

**arr-coc-0-1 production GPU security:**
```yaml
# arr-coc-0-1 security architecture
arr_coc_security:
  project: "arr-coc-0-1"
  compliance: "SOC 2"
  data_classification: "Proprietary (non-PHI)"

  infrastructure:
    gpu_instances:
      machine_type: "a2-highgpu-8g"
      gpu: "NVIDIA A100 80GB x8"
      shielded_vm:
        secure_boot: enabled
        vtpm: enabled
        integrity_monitoring: enabled
      encryption:
        boot_disk: "CMEK"
        persistent_disk: "CMEK"
        gcs_buckets: "CMEK"

    networking:
      vpc: "arr-coc-vpc (isolated)"
      subnet: "ml-training-subnet (10.128.0.0/20)"
      external_ip: disabled
      private_google_access: enabled
      vpc_service_controls: enabled

    access_control:
      authentication: "Cloud Identity + MFA"
      service_account: "gpu-training-sa@arr-coc-0-1.iam.gserviceaccount.com"
      iam_roles: ["roles/aiplatform.user", "roles/storage.objectViewer"]

    monitoring:
      audit_logging: "Full (Admin + Data Access)"
      dashboards:
        - "GPU Security Dashboard"
        - "SOC 2 Compliance Dashboard"
      alerts:
        - "Unprotected GPU instance creation"
        - "External IP assignment"
        - "Integrity violation"
```

### 6.2 Security Validation Checklist

**Pre-production validation:**
```bash
#!/bin/bash
# arr-coc-0-1: Security validation checklist

PROJECT_ID="arr-coc-0-1"
ZONE="us-west2-b"
INSTANCE_NAME="arr-coc-gpu-training"

echo "=== arr-coc-0-1 GPU Security Validation ==="

# 1. Verify Shielded VM enabled
echo -n "1. Shielded VM enabled: "
VTPM=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --format="value(shieldedInstanceConfig.enableVtpm)" \
    --project=$PROJECT_ID)
[[ "$VTPM" == "True" ]] && echo "PASS" || echo "FAIL"

# 2. Verify no external IP
echo -n "2. No external IP: "
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)" \
    --project=$PROJECT_ID)
[[ -z "$EXTERNAL_IP" ]] && echo "PASS" || echo "FAIL (IP: $EXTERNAL_IP)"

# 3. Verify CMEK encryption
echo -n "3. CMEK encryption: "
CMEK=$(gcloud compute disks describe $INSTANCE_NAME \
    --zone=$ZONE \
    --format="value(diskEncryptionKey.kmsKeyName)" \
    --project=$PROJECT_ID)
[[ "$CMEK" =~ "arr-coc-data-key" ]] && echo "PASS" || echo "FAIL"

# 4. Verify service account
echo -n "4. Dedicated service account: "
SA=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --format="value(serviceAccounts[0].email)" \
    --project=$PROJECT_ID)
[[ "$SA" == "gpu-training-sa@${PROJECT_ID}.iam.gserviceaccount.com" ]] && echo "PASS" || echo "FAIL"

# 5. Verify compliance labels
echo -n "5. Compliance labels: "
LABELS=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --format="value(labels.compliance)" \
    --project=$PROJECT_ID)
[[ "$LABELS" == "soc2" ]] && echo "PASS" || echo "FAIL"

# 6. Verify audit logging enabled
echo -n "6. Audit logging enabled: "
AUDIT=$(gcloud projects get-iam-policy $PROJECT_ID --format=json | \
    jq '.auditConfigs[] | select(.service=="compute.googleapis.com") | .auditLogConfigs[] | select(.logType=="DATA_READ")')
[[ -n "$AUDIT" ]] && echo "PASS" || echo "FAIL"

echo "=== Validation Complete ==="
```

---

## Summary

**GPU security and compliance best practices:**

1. **Hardware Security**: Enable Shielded VM (Secure Boot, vTPM, Integrity Monitoring) on all production GPU instances
2. **Confidential Computing**: Use H100 GPUs with Confidential VMs for highly sensitive data (PHI, PII)
3. **Network Isolation**: Deploy GPUs in isolated VPCs with no external IPs, use VPC Service Controls for perimeter security
4. **Workload Isolation**: Use MIG (Multi-Instance GPU) for multi-tenant GPU workloads
5. **Encryption**: Enable CMEK for boot disks, persistent disks, and GCS buckets
6. **Compliance**: Sign BAAs for HIPAA, implement tokenization for PCI-DSS, maintain SOC 2 controls
7. **Audit Logging**: Enable comprehensive audit logs (Admin + Data Access), retain for compliance requirements
8. **Monitoring**: Create security dashboards, configure real-time alerts for violations

**arr-coc-0-1 security posture:**
- Shielded VMs with all features enabled
- CMEK encryption for all data
- VPC isolation with no external IPs
- SOC 2 compliant architecture
- Comprehensive audit logging and monitoring

---

## Sources

**Source Documents:**
- [gcp-vertex/16-vpc-service-controls-private.md](../gcp-vertex/16-vpc-service-controls-private.md) - VPC Service Controls, Private Google Access, PSC architecture
- [gcp-vertex/18-compliance-governance-audit.md](../gcp-vertex/18-compliance-governance-audit.md) - HIPAA, PCI-DSS, SOC 2 compliance frameworks
- [gcloud-iam/00-service-accounts-ml-security.md](../gcloud-iam/00-service-accounts-ml-security.md) - IAM best practices, service account security

**Web Research:**
- [Google Cloud Shielded VM](https://docs.cloud.google.com/compute/shielded-vm/docs/shielded-vm) (accessed 2025-11-16)
- [Google Cloud Confidential Computing](https://cloud.google.com/security/products/confidential-computing) (accessed 2025-11-16)
- [Google Cloud Confidential VM Release Notes](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/release-notes) (accessed 2025-11-16)
- [Google Cloud HIPAA Compliance](https://docs.cloud.google.com/security/compliance/hipaa) (accessed 2025-11-16)
- [How Confidential Accelerators Boost AI Workload Security](https://cloud.google.com/blog/products/identity-security/how-confidential-accelerators-can-boost-ai-workload-security) (accessed 2025-11-16)
- [Security Google Cloud Community - Confidential GPUs](https://security.googlecloudcommunity.com/community-blog-42/protecting-your-data-why-confidential-computing-is-necessary-for-your-business-4007) (accessed 2025-11-16)
- [Massed Compute FAQ - Shielded VMs with GPUs](https://massedcompute.com/faq-answers/?question=Can%20I%20use%20Shielded%20VMs%20with%20NVIDIA%20data%20center%20GPUs%20in%20Google%20Cloud?) (accessed 2025-11-16)
- [Massed Compute FAQ - Securing NVIDIA Cloud Services](https://massedcompute.com/faq-answers/?question=What+are+the+best+practices+for+securing+NVIDIA+cloud+services%3F) (accessed 2025-11-16)
- [DevZero GPU Security and Isolation](https://www.devzero.io/blog/gpu-security-and-isolation) (accessed 2025-11-16)
- [Phala Confidential Computing Trends 2025](https://phala.com/learn/confidential-computing-trends-2025) (accessed 2025-11-16)
- [Corvex AI HIPAA GPU Providers](https://www.corvex.ai/blog/top-hipaa-compliant-cloud-gpu-providers-for-secure-ai-model-training) (accessed 2025-11-16)
- [NexGen Cloud GPU Security Best Practices](https://www.nexgencloud.com/blog/thought-leadership/top-5-best-practices-for-securing-gpu-accelerated-ai-cloud-environments) (accessed 2025-11-16)

**Additional References:**
- Google search: "Shielded VM GPU instances GCP 2024 2025" (accessed 2025-11-16)
- Google search: "Confidential Computing GPU support GCP 2024 2025" (accessed 2025-11-16)
- Google search: "GPU workload isolation security GCP best practices 2024" (accessed 2025-11-16)
- Google search: "HIPAA compliant GPU training GCP compliance 2024 2025" (accessed 2025-11-16)
