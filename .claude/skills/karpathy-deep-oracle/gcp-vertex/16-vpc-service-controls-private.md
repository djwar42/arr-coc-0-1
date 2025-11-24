# VPC Service Controls & Private Networking for Vertex AI

## Overview

VPC Service Controls (VPC-SC) and private networking configurations provide enterprise-grade data exfiltration protection and network isolation for Vertex AI workloads. These security controls are essential for HIPAA, PCI-DSS, and SOC 2 compliance requirements.

**Core Security Principle**: VPC Service Controls create a security perimeter around Google Cloud resources to prevent unauthorized data access and exfiltration, while private networking ensures ML workloads never traverse the public internet.

From [VPC Service Controls Overview](https://docs.cloud.google.com/vpc-service-controls/docs/overview) (accessed 2025-11-16):
> "VPC Service Controls lets you define security policies that prevent access to Google-managed services outside of a trusted perimeter, block access to data from unauthorized networks, and mitigate data exfiltration risks."

---

## 1. VPC Service Controls Architecture

### 1.1 Service Perimeter Fundamentals

**Service Perimeter** = Security boundary around Google Cloud resources

```yaml
# Service Perimeter Configuration
apiVersion: accesscontextmanager.cnrm.cloud.google.com/v1beta1
kind: AccessContextManagerServicePerimeter
metadata:
  name: vertex-ai-perimeter
spec:
  title: "Vertex AI ML Training Perimeter"
  perimeterType: PERIMETER_TYPE_REGULAR
  status:
    resources:
      - "projects/123456789"  # ML training project
    restrictedServices:
      - "aiplatform.googleapis.com"
      - "storage.googleapis.com"
      - "bigquery.googleapis.com"
    vpcAccessibleServices:
      enableRestriction: true
      allowedServices:
        - "aiplatform.googleapis.com"
        - "storage.googleapis.com"
```

**Protected Resources**:
- Vertex AI training jobs
- GCS buckets (datasets, checkpoints, logs)
- BigQuery datasets
- Artifact Registry repositories
- Cloud Storage Transfer Service

From [Vertex AI VPC Service Controls](https://docs.cloud.google.com/vertex-ai/docs/general/vpc-service-controls) (accessed 2025-11-16):
> "When you use VPC Service Controls to protect Vertex AI resources, you create a service perimeter that protects the resources and data that you specify."

### 1.2 Ingress Rules

**Ingress rules** control traffic INTO the perimeter from outside networks.

```python
# Ingress Rule: Allow from Trusted Corporate Network
from google.cloud import accesscontextmanager_v1

ingress_policy = {
    "ingress_from": {
        "sources": [
            {
                "access_level": "accessPolicies/123/accessLevels/trusted_corp_network"
            }
        ],
        "identities": [
            "user:data-scientist@company.com",
            "serviceAccount:ml-pipeline@project.iam.gserviceaccount.com"
        ]
    },
    "ingress_to": {
        "resources": ["*"],
        "operations": [
            {
                "service_name": "aiplatform.googleapis.com",
                "method_selectors": [
                    {"method": "CreateCustomJob"},
                    {"method": "GetCustomJob"},
                    {"method": "CreateModelDeploymentMonitoringJob"}
                ]
            },
            {
                "service_name": "storage.googleapis.com",
                "method_selectors": [
                    {"method": "google.storage.objects.get"},
                    {"method": "google.storage.objects.create"}
                ]
            }
        ]
    }
}
```

**Common Ingress Patterns**:

1. **On-Premise Access**:
```yaml
ingress_from:
  sources:
    - resource: "//compute.googleapis.com/projects/PROJECT/regions/us-central1/subnetworks/on-prem-vpn"
  identities:
    - "serviceAccount:on-prem-sync@project.iam.gserviceaccount.com"
```

2. **CI/CD Pipeline Access**:
```yaml
ingress_from:
  sources:
    - access_level: "accessPolicies/123/accessLevels/cicd_runners"
  identities:
    - "serviceAccount:github-actions@project.iam.gserviceaccount.com"
```

3. **Cross-Project Collaboration**:
```yaml
ingress_from:
  sources:
    - resource: "projects/456"  # Research project
  identities:
    - "group:ml-researchers@company.com"
```

From [Ingress and Egress Rules](https://docs.cloud.google.com/vpc-service-controls/docs/ingress-egress-rules) (accessed 2025-11-16):
> "VPC Service Controls uses ingress and egress rules to allow access to and from the resources and clients protected by service perimeters."

### 1.3 Egress Rules

**Egress rules** control traffic OUT of the perimeter to external services.

```python
# Egress Rule: Allow to Approved External Registry
egress_policy = {
    "egress_from": {
        "identities": [
            "serviceAccount:training-job@project.iam.gserviceaccount.com"
        ]
    },
    "egress_to": {
        "resources": [
            "projects/external-registry-project"
        ],
        "operations": [
            {
                "service_name": "artifactregistry.googleapis.com",
                "method_selectors": [
                    {"method": "docker.pulls"}  # Read-only access
                ]
            }
        ]
    }
}
```

**Data Exfiltration Prevention Examples**:

1. **Block All External BigQuery Exports**:
```yaml
# NO egress rule for BigQuery → prevents data export
egress_to:
  resources: []  # Empty = no external access
  operations: []
```

2. **Allow Approved Model Registry Only**:
```yaml
egress_to:
  resources:
    - "projects/approved-model-hub"
  operations:
    - service_name: "aiplatform.googleapis.com"
      method_selectors:
        - method: "UploadModel"  # Can upload trained models
        # NOTE: GetModel NOT allowed → can't download external models
```

3. **Controlled Logging Export**:
```yaml
egress_to:
  resources:
    - "projects/central-logging"
  operations:
    - service_name: "logging.googleapis.com"
      method_selectors:
        - method: "WriteLogEntries"
```

**Security Best Practice**: Default-deny egress, explicitly allow only necessary external access.

---

## 2. Private Google Access

### 2.1 Enabling Private Google Access

**Private Google Access** allows VMs with private IPs to reach Google APIs without public IP addresses.

```bash
# Enable Private Google Access on subnet
gcloud compute networks subnets update training-subnet \
    --region=us-central1 \
    --enable-private-ip-google-access

# Verify configuration
gcloud compute networks subnets describe training-subnet \
    --region=us-central1 \
    --format="get(privateIpGoogleAccess)"
```

**Architecture**:
```
Vertex AI Training Job (10.128.0.5 - private IP)
    ↓
Private Google Access route (199.36.153.8/30)
    ↓
Google APIs (aiplatform.googleapis.com)
    ↓
No public internet traversal
```

### 2.2 Custom Routes for Private Access

```bash
# Create custom route to Google APIs
gcloud compute routes create private-google-access \
    --network=custom-vpc \
    --destination-range=199.36.153.8/30 \
    --next-hop-gateway=default-internet-gateway \
    --priority=1000

# Create route to restricted.googleapis.com (VPC-SC compatible)
gcloud compute routes create restricted-google-apis \
    --network=custom-vpc \
    --destination-range=199.36.153.4/30 \
    --next-hop-gateway=default-internet-gateway \
    --priority=1000
```

**DNS Configuration**:
```bash
# Use restricted.googleapis.com for VPC-SC perimeters
gcloud dns managed-zones create restricted-google-apis \
    --dns-name="googleapis.com." \
    --description="Route to restricted Google APIs" \
    --networks=custom-vpc \
    --visibility=private

# Add CNAME record
gcloud dns record-sets create "*.googleapis.com." \
    --zone="restricted-google-apis" \
    --type=CNAME \
    --ttl=300 \
    --rrdatas="restricted.googleapis.com."
```

From [Private Google Access Configuration](https://cloud.google.com/vpc/docs/configure-private-google-access) (accessed 2025-11-16):
> "Private Google Access enables VM instances that only have internal IP addresses to reach Google APIs and services."

### 2.3 arr-coc-0-1 Private Access Configuration

```python
# arr-coc-0-1: Vertex AI Custom Job with Private Google Access
from google.cloud import aiplatform

aiplatform.init(
    project="arr-coc-training",
    location="us-west2"
)

# Custom Job configured for private access
worker_pool_spec = {
    "machine_spec": {
        "machine_type": "a2-highgpu-8g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 8
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": "us-west2-docker.pkg.dev/arr-coc-training/training/arr-coc:latest",
        "args": [
            "--dataset=gs://arr-coc-data/train",
            "--checkpoint-dir=gs://arr-coc-checkpoints",
            "--wandb-project=arr-coc-ablations"
        ]
    },
    # Private networking configuration
    "network": "projects/arr-coc-training/global/networks/ml-vpc",
    "reserved_ip_ranges": [],  # Use subnet's IP range
    "enable_web_access": False,  # No public internet
    "service_account": "training-job@arr-coc-training.iam.gserviceaccount.com"
}

job = aiplatform.CustomJob(
    display_name="arr-coc-private-training",
    worker_pool_specs=[worker_pool_spec],
    encryption_spec_key_name="projects/arr-coc-training/locations/us-west2/keyRings/ml-training/cryptoKeys/data-key"
)

job.run()
```

---

## 3. Private Service Connect (PSC)

### 3.1 PSC Endpoints for Vertex AI

**Private Service Connect** provides private connectivity to Vertex AI without VPC Peering overhead.

```bash
# Create PSC endpoint for Vertex AI
gcloud compute addresses create vertex-ai-psc \
    --region=us-central1 \
    --subnet=training-subnet \
    --addresses=10.128.0.100

# Create PSC forwarding rule
gcloud compute forwarding-rules create vertex-ai-endpoint \
    --region=us-central1 \
    --network=custom-vpc \
    --address=vertex-ai-psc \
    --target-service-attachment=projects/cloud-ai-platform-public/regions/us-central1/serviceAttachments/vertex-ai-apis \
    --service-directory-registration=projects/PROJECT_ID/locations/us-central1
```

**Network Attachment (for Shared VPC)**:
```bash
# Create network attachment in service project
gcloud beta compute network-attachments create vertex-psc-attachment \
    --region=us-central1 \
    --subnets=training-subnet \
    --connection-preference=ACCEPT_AUTOMATIC \
    --producer-accept-list=cloud-ai-platform-public \
    --project=ml-training-service-project
```

### 3.2 PSC vs VPC Peering Comparison

| Feature | VPC Peering | Private Service Connect |
|---------|-------------|-------------------------|
| **IP Range Consumption** | Large (/16 required) | Small (/28 sufficient) |
| **Routing Complexity** | Transitive routing issues | Isolated per endpoint |
| **Multi-Region** | Complex (peering per region) | Simple (endpoint per region) |
| **IP Range Conflicts** | High risk | Low risk |
| **Bandwidth** | High (direct connection) | High (optimized path) |
| **Setup Complexity** | Medium | Low |
| **Cost** | Peering charges | PSC endpoint charges |

From [gcp-vertex/00-custom-jobs-advanced.md](../gcp-vertex/00-custom-jobs-advanced.md) (lines 286-299):
> "Private Service Connect provides private connectivity without VPC Peering. PSC vs VPC Peering: IP Range Consumption - VPC Peering requires Large (/16), PSC requires Small (/28); Routing Complexity - VPC Peering has transitive routing issues, PSC has isolated, no transitive routing."

**Recommendation**: Use PSC for new deployments; migrate from VPC Peering to PSC for reduced IP consumption.

### 3.3 PSC Configuration for Vertex AI Pipelines

```python
# Vertex AI Pipelines with PSC
from kfp import dsl
from google.cloud import aiplatform

@dsl.pipeline(name="arr-coc-training-pipeline")
def training_pipeline(
    dataset_uri: str,
    model_uri: str
):
    preprocess_op = dsl.ContainerOp(
        name="preprocess",
        image="us-west2-docker.pkg.dev/arr-coc/training/preprocess:latest",
        arguments=["--input", dataset_uri]
    )

    train_op = dsl.ContainerOp(
        name="train",
        image="us-west2-docker.pkg.dev/arr-coc/training/train:latest",
        arguments=["--data", preprocess_op.output]
    )

# Compile and run with PSC
aiplatform.init(
    project="arr-coc-training",
    location="us-west2",
    # PSC endpoint for Pipelines API
    api_endpoint="10.128.0.100"  # Private IP
)

job = aiplatform.PipelineJob(
    display_name="arr-coc-psc-pipeline",
    template_path="pipeline.json",
    parameter_values={
        "dataset_uri": "gs://arr-coc-data/train",
        "model_uri": "gs://arr-coc-models"
    },
    encryption_spec_key_name="projects/arr-coc-training/locations/us-west2/keyRings/ml/cryptoKeys/pipeline"
)

job.run(network="projects/arr-coc-training/global/networks/ml-vpc")
```

From [Private Service Connect Interface Vertex AI Pipelines](https://codelabs.developers.google.com/psc-interface-pipelines) (accessed 2025-11-16):
> "In this tutorial you'll learn how to configure and validate the Private Service Connect Vertex AI Pipelines."

---

## 4. Shared VPC Configuration

### 4.1 Host Project Setup

**Shared VPC** allows centralized network management across multiple projects.

```bash
# Enable Shared VPC in host project
gcloud compute shared-vpc enable network-host-project

# Create VPC network in host project
gcloud compute networks create ml-shared-vpc \
    --subnet-mode=custom \
    --project=network-host-project

# Create subnet for ML workloads
gcloud compute networks subnets create ml-training-subnet \
    --network=ml-shared-vpc \
    --region=us-central1 \
    --range=10.128.0.0/20 \
    --enable-private-ip-google-access \
    --enable-flow-logs \
    --project=network-host-project
```

### 4.2 Service Project Attachment

```bash
# Attach service project to shared VPC
gcloud compute shared-vpc associated-projects add ml-training-dev \
    --host-project=network-host-project

gcloud compute shared-vpc associated-projects add ml-training-prod \
    --host-project=network-host-project

# Grant Vertex AI service agent network user role
gcloud projects add-iam-policy-binding network-host-project \
    --member="serviceAccount:service-123456789@gcp-sa-aiplatform.iam.gserviceaccount.com" \
    --role="roles/compute.networkUser"

gcloud projects add-iam-policy-binding network-host-project \
    --member="serviceAccount:service-123456789@gcp-sa-aiplatform.iam.gserviceaccount.com" \
    --role="roles/compute.securityAdmin"
```

### 4.3 Vertex AI in Shared VPC

```python
# Service project: ml-training-prod
from google.cloud import aiplatform

aiplatform.init(
    project="ml-training-prod",
    location="us-central1"
)

# Reference host project's shared VPC
job = aiplatform.CustomJob(
    display_name="shared-vpc-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-8g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 8
        },
        "replica_count": 2,
        "container_spec": {
            "image_uri": "gcr.io/ml-training-prod/training:latest"
        },
        # Reference host project's network
        "network": "projects/network-host-project/global/networks/ml-shared-vpc"
    }]
)

job.run()
```

**Shared VPC Architecture**:
```
Host Project: network-host-project
├── Shared VPC: ml-shared-vpc
│   ├── Subnet: ml-training-subnet (us-central1, 10.128.0.0/20)
│   ├── Subnet: ml-inference-subnet (us-east1, 10.129.0.0/20)
│   └── Firewall: allow-ml-internal
└── Service Projects
    ├── ml-training-dev (uses ml-training-subnet)
    ├── ml-training-prod (uses ml-training-subnet)
    └── ml-inference-prod (uses ml-inference-subnet)
```

From search results on Shared VPC Vertex AI configuration (accessed 2025-11-16):
> "When configuring Vertex AI with a Shared VPC, create the network attachment in the service project where you use Vertex AI. This approach helps prevent authorization issues."

---

## 5. Firewall Rules for Vertex AI

### 5.1 Essential Firewall Rules

```bash
# Allow internal Vertex AI communication
gcloud compute firewall-rules create allow-vertex-internal \
    --network=ml-shared-vpc \
    --allow=tcp:0-65535,udp:0-65535,icmp \
    --source-ranges=10.128.0.0/20 \
    --target-tags=vertex-ai-training \
    --priority=1000 \
    --project=network-host-project

# Allow health checks from Google
gcloud compute firewall-rules create allow-health-checks \
    --network=ml-shared-vpc \
    --allow=tcp:80,tcp:443 \
    --source-ranges=35.191.0.0/16,130.211.0.0/22 \
    --target-tags=vertex-ai-endpoint \
    --priority=1000 \
    --project=network-host-project

# Allow SSH from IAP (for debugging)
gcloud compute firewall-rules create allow-iap-ssh \
    --network=ml-shared-vpc \
    --allow=tcp:22 \
    --source-ranges=35.235.240.0/20 \
    --target-tags=vertex-ai-training \
    --priority=1000 \
    --project=network-host-project
```

### 5.2 Distributed Training Firewall Rules

```bash
# Allow NCCL communication for multi-GPU training
gcloud compute firewall-rules create allow-nccl \
    --network=ml-shared-vpc \
    --allow=tcp:1024-65535 \
    --source-tags=vertex-ai-training \
    --target-tags=vertex-ai-training \
    --priority=900 \
    --description="NCCL all-reduce for multi-GPU training" \
    --project=network-host-project

# Allow TensorFlow distributed training
gcloud compute firewall-rules create allow-tf-dist \
    --network=ml-shared-vpc \
    --allow=tcp:2222 \
    --source-tags=vertex-ai-training \
    --target-tags=vertex-ai-training \
    --priority=900 \
    --description="TensorFlow parameter server communication" \
    --project=network-host-project

# Allow PyTorch DDP communication
gcloud compute firewall-rules create allow-pytorch-ddp \
    --network=ml-shared-vpc \
    --allow=tcp:29500-29599 \
    --source-tags=vertex-ai-training \
    --target-tags=vertex-ai-training \
    --priority=900 \
    --description="PyTorch DistributedDataParallel rendezvous" \
    --project=network-host-project
```

### 5.3 Deny Rules for Security

```bash
# Deny all egress to public internet (except approved Google APIs)
gcloud compute firewall-rules create deny-all-egress \
    --network=ml-shared-vpc \
    --action=DENY \
    --rules=all \
    --destination-ranges=0.0.0.0/0 \
    --priority=65534 \
    --direction=EGRESS \
    --project=network-host-project

# Allow egress to Google APIs only
gcloud compute firewall-rules create allow-google-apis \
    --network=ml-shared-vpc \
    --action=ALLOW \
    --rules=tcp:443 \
    --destination-ranges=199.36.153.8/30,199.36.153.4/30 \
    --priority=100 \
    --direction=EGRESS \
    --project=network-host-project
```

**Firewall Rule Priority**:
- 0-999: High priority (allow specific traffic)
- 1000-64999: Normal priority (standard rules)
- 65000-65535: Low priority (deny-all fallback)

---

## 6. Cloud DLP Integration

### 6.1 DLP for Data Discovery

**Cloud Data Loss Prevention (DLP)** scans datasets for sensitive information (PII, PHI, PCI).

```python
# Scan GCS bucket for PII before training
from google.cloud import dlp_v2

dlp_client = dlp_v2.DlpServiceClient()

# Define inspection configuration
inspect_config = {
    "info_types": [
        {"name": "EMAIL_ADDRESS"},
        {"name": "PHONE_NUMBER"},
        {"name": "CREDIT_CARD_NUMBER"},
        {"name": "US_SOCIAL_SECURITY_NUMBER"},
        {"name": "MEDICAL_RECORD_NUMBER"}
    ],
    "min_likelihood": "LIKELY",
    "limits": {"max_findings_per_request": 100}
}

# Scan GCS bucket
storage_config = {
    "cloud_storage_options": {
        "file_set": {
            "url": "gs://arr-coc-data/train/**"
        }
    }
}

# Create DLP scan job
parent = f"projects/{project_id}/locations/us"
job_config = {
    "inspect_config": inspect_config,
    "storage_config": storage_config,
    "actions": [
        {
            "save_findings": {
                "output_config": {
                    "table": {
                        "project_id": project_id,
                        "dataset_id": "dlp_results",
                        "table_id": "pii_findings"
                    }
                }
            }
        }
    ]
}

dlp_job = dlp_client.create_dlp_job(
    request={"parent": parent, "inspect_job": job_config}
)

print(f"DLP Job created: {dlp_job.name}")
```

### 6.2 De-identification for Training

```python
# De-identify dataset before training
deidentify_config = {
    "info_type_transformations": {
        "transformations": [
            {
                "primitive_transformation": {
                    "crypto_replace_ffx_fpe_config": {
                        "crypto_key": {
                            "kms_wrapped": {
                                "wrapped_key": kms_wrapped_key,
                                "crypto_key_name": "projects/PROJECT/locations/us/keyRings/dlp/cryptoKeys/de-id"
                            }
                        },
                        "common_alphabet": "ALPHA_NUMERIC"
                    }
                },
                "info_types": [
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "PHONE_NUMBER"}
                ]
            },
            {
                "primitive_transformation": {
                    "redact_config": {}  # Redact SSNs completely
                },
                "info_types": [
                    {"name": "US_SOCIAL_SECURITY_NUMBER"}
                ]
            }
        ]
    }
}

# Apply de-identification
deidentify_response = dlp_client.deidentify_content(
    request={
        "parent": parent,
        "deidentify_config": deidentify_config,
        "item": {"table": input_table}
    }
)

# Use de-identified data for training
deidentified_data = deidentify_response.item.table
```

### 6.3 DLP + VPC-SC Integration

```python
# DLP scanning within VPC-SC perimeter
from google.cloud import dlp_v2, storage

# DLP operates INSIDE perimeter (no data exfiltration)
dlp_client = dlp_v2.DlpServiceClient()

# Scan bucket within perimeter
inspect_job = dlp_client.create_dlp_job(
    request={
        "parent": f"projects/{project_id}/locations/us",
        "inspect_job": {
            "storage_config": {
                "cloud_storage_options": {
                    "file_set": {"url": "gs://perimeter-protected-data/train/**"}
                }
            },
            "inspect_config": inspect_config,
            # Save findings WITHIN perimeter
            "actions": [{
                "save_findings": {
                    "output_config": {
                        "table": {
                            "project_id": project_id,  # Same perimeter
                            "dataset_id": "dlp_findings",
                            "table_id": f"scan_{job_id}"
                        }
                    }
                }
            }]
        }
    }
)
```

**DLP + VPC-SC Benefits**:
- DLP scanning happens inside perimeter
- Findings stay within security boundary
- No sensitive data crosses perimeter
- Automated compliance verification

From search results on Cloud DLP integration (accessed 2025-11-16):
> "Cloud DLP integration with VPC Service Controls ensures that sensitive data scanning and findings remain within the security perimeter, preventing data exfiltration while enabling compliance monitoring."

---

## 7. Compliance Requirements

### 7.1 HIPAA Compliance Configuration

**HIPAA** (Health Insurance Portability and Accountability Act) requires:
- Encryption at rest and in transit
- Access controls and audit logging
- Data isolation and perimeter security

```bash
# Enable HIPAA-compliant configuration
# 1. Enable Cloud Audit Logs (required for HIPAA)
gcloud logging sinks create hipaa-audit-logs \
    bigquery.googleapis.com/projects/PROJECT_ID/datasets/audit_logs \
    --log-filter='protoPayload.serviceName="aiplatform.googleapis.com"'

# 2. Enable VPC-SC perimeter
gcloud access-context-manager perimeters create hipaa-ml-perimeter \
    --title="HIPAA ML Training Perimeter" \
    --resources="projects/123456789" \
    --restricted-services="aiplatform.googleapis.com,storage.googleapis.com,bigquery.googleapis.com" \
    --policy=POLICY_ID

# 3. Enable CMEK (Customer-Managed Encryption Keys)
gcloud kms keyrings create ml-training \
    --location=us-central1

gcloud kms keys create data-encryption-key \
    --location=us-central1 \
    --keyring=ml-training \
    --purpose=encryption

# 4. Grant Vertex AI service agent access to KMS key
gcloud kms keys add-iam-policy-binding data-encryption-key \
    --location=us-central1 \
    --keyring=ml-training \
    --member="serviceAccount:service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

**HIPAA Training Job**:
```python
# Vertex AI Custom Job with HIPAA compliance
job = aiplatform.CustomJob(
    display_name="hipaa-compliant-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "gcr.io/PROJECT/hipaa-training:latest"
        },
        # HIPAA requirements
        "network": "projects/PROJECT/global/networks/hipaa-vpc",  # Isolated VPC
        "enable_web_access": False,  # No public internet
        "service_account": "hipaa-training@PROJECT.iam.gserviceaccount.com"
    }],
    # CMEK encryption
    encryption_spec_key_name="projects/PROJECT/locations/us-central1/keyRings/ml-training/cryptoKeys/data-encryption-key",
    # Labels for audit
    labels={
        "compliance": "hipaa",
        "data-classification": "phi",
        "environment": "production"
    }
)
```

From [HIPAA Compliance on Google Cloud](https://docs.cloud.google.com/security/compliance/hipaa) (accessed 2025-11-16):
> "Google Cloud services covered under HIPAA compliance include Vertex AI, Cloud Storage, BigQuery, and VPC Service Controls when used within a signed Business Associate Agreement (BAA)."

### 7.2 PCI-DSS Compliance Configuration

**PCI-DSS** (Payment Card Industry Data Security Standard) requires:
- Network segmentation
- Encrypted transmission of cardholder data
- Access controls and monitoring

```python
# PCI-DSS compliant Vertex AI configuration
from google.cloud import aiplatform

# Separate VPC for PCI workloads
pci_network_config = {
    "network": "projects/PROJECT/global/networks/pci-isolated-vpc",
    "reserved_ip_ranges": [],
    "enable_web_access": False
}

# Create training job in PCI-compliant environment
job = aiplatform.CustomJob(
    display_name="pci-dss-fraud-detection",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "n1-highmem-16",
            "accelerator_type": "NVIDIA_TESLA_V100",
            "accelerator_count": 2
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "gcr.io/PROJECT/pci-fraud-model:latest",
            "env": [
                {"name": "TOKENIZATION_KEY", "value": "projects/PROJECT/secrets/tokenization-key/versions/latest"}
            ]
        },
        **pci_network_config
    }],
    encryption_spec_key_name="projects/PROJECT/locations/us/keyRings/pci/cryptoKeys/cardholder-data",
    labels={
        "compliance": "pci-dss",
        "cardholder-data": "yes",
        "environment": "production"
    }
)

# VPC-SC perimeter for PCI-DSS
pci_perimeter_config = {
    "resources": ["projects/PCI_PROJECT"],
    "restricted_services": [
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "secretmanager.googleapis.com"
    ],
    # NO ingress/egress except approved paths
    "ingress_policies": [
        {
            "ingress_from": {
                "sources": [{"access_level": "accessPolicies/POLICY/accessLevels/pci_secure_zone"}],
                "identities": ["group:pci-authorized-users@company.com"]
            },
            "ingress_to": {
                "resources": ["*"],
                "operations": [
                    {"service_name": "aiplatform.googleapis.com", "method_selectors": [{"method": "*"}]}
                ]
            }
        }
    ],
    "egress_policies": []  # No egress = data cannot leave perimeter
}
```

**PCI-DSS Firewall Rules**:
```bash
# Segment PCI network from rest of organization
gcloud compute firewall-rules create deny-pci-egress \
    --network=pci-isolated-vpc \
    --action=DENY \
    --rules=all \
    --destination-ranges=0.0.0.0/0 \
    --priority=100 \
    --direction=EGRESS

# Allow only to approved services
gcloud compute firewall-rules create allow-pci-vertex-ai \
    --network=pci-isolated-vpc \
    --action=ALLOW \
    --rules=tcp:443 \
    --destination-ranges=199.36.153.8/30 \
    --priority=50 \
    --direction=EGRESS
```

From search results on PCI-DSS compliance (accessed 2025-11-16):
> "Google Cloud's PCI DSS certification meets the PCI DSS 4.0.1 compliance standard. VPC Service Controls, network segmentation, and encryption are essential for protecting cardholder data."

### 7.3 SOC 2 Compliance Configuration

**SOC 2** (Service Organization Control 2) focuses on:
- Security
- Availability
- Processing integrity
- Confidentiality
- Privacy

```yaml
# SOC 2 Compliance Checklist for Vertex AI
soc2_controls:
  security:
    - VPC Service Controls enabled: ✓
    - Encryption at rest (CMEK): ✓
    - Encryption in transit (TLS 1.2+): ✓
    - Multi-factor authentication: ✓
    - IAM least privilege: ✓

  availability:
    - Multi-region deployment: ✓
    - Automated backups: ✓
    - Disaster recovery plan: ✓
    - SLA monitoring: ✓

  processing_integrity:
    - Data validation: ✓
    - Model versioning: ✓
    - Audit trails: ✓
    - Change management: ✓

  confidentiality:
    - VPC-SC perimeter: ✓
    - Private Google Access: ✓
    - DLP scanning: ✓
    - Access reviews: ✓

  privacy:
    - Data minimization: ✓
    - Retention policies: ✓
    - Right to deletion: ✓
    - Privacy impact assessment: ✓
```

**Audit Logging for SOC 2**:
```bash
# Enable comprehensive audit logging
gcloud logging sinks create soc2-audit-trail \
    bigquery.googleapis.com/projects/PROJECT/datasets/soc2_audit \
    --log-filter='
        protoPayload.serviceName="aiplatform.googleapis.com" OR
        protoPayload.serviceName="storage.googleapis.com" OR
        protoPayload.serviceName="iam.googleapis.com" OR
        protoPayload.serviceName="accesscontextmanager.googleapis.com"
    '

# Enable Data Access logs (normally disabled)
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:soc2-auditor@PROJECT.iam.gserviceaccount.com" \
    --role="roles/logging.privateLogViewer"

# Create dashboard for SOC 2 compliance monitoring
gcloud monitoring dashboards create --config-from-file=soc2-dashboard.yaml
```

---

## 8. arr-coc-0-1 Production Security Configuration

### 8.1 Complete VPC-SC Setup

```bash
# arr-coc-0-1 production security configuration

# 1. Create VPC-SC perimeter
gcloud access-context-manager perimeters create arr-coc-prod-perimeter \
    --title="ARR-COC Production ML Training" \
    --resources="projects/arr-coc-training" \
    --restricted-services="aiplatform.googleapis.com,storage.googleapis.com,artifactregistry.googleapis.com" \
    --access-levels="accessPolicies/POLICY/accessLevels/ml_researchers" \
    --policy=POLICY_ID

# 2. Enable Private Google Access
gcloud compute networks subnets update ml-training-subnet \
    --region=us-west2 \
    --enable-private-ip-google-access \
    --project=arr-coc-training

# 3. Create PSC endpoint
gcloud compute addresses create arr-coc-vertex-psc \
    --region=us-west2 \
    --subnet=ml-training-subnet \
    --addresses=10.128.0.200

gcloud compute forwarding-rules create arr-coc-vertex-endpoint \
    --region=us-west2 \
    --network=arr-coc-vpc \
    --address=arr-coc-vertex-psc \
    --target-service-attachment=projects/cloud-ai-platform-public/regions/us-west2/serviceAttachments/vertex-ai-apis

# 4. Configure firewall rules
gcloud compute firewall-rules create arr-coc-allow-nccl \
    --network=arr-coc-vpc \
    --allow=tcp:1024-65535 \
    --source-tags=arr-coc-training \
    --target-tags=arr-coc-training \
    --priority=900

gcloud compute firewall-rules create arr-coc-deny-internet \
    --network=arr-coc-vpc \
    --action=DENY \
    --rules=all \
    --destination-ranges=0.0.0.0/0 \
    --priority=65534 \
    --direction=EGRESS

# 5. Enable DLP scanning
gcloud services enable dlp.googleapis.com --project=arr-coc-training
```

### 8.2 Secure Training Pipeline

```python
# arr-coc-0-1: Production-ready secure training pipeline
from google.cloud import aiplatform
from google.cloud import dlp_v2

class SecureARRCOCTrainer:
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location

        aiplatform.init(
            project=project_id,
            location=location,
            # Use PSC endpoint
            api_endpoint="10.128.0.200"
        )

        self.dlp_client = dlp_v2.DlpServiceClient()

    def scan_dataset(self, dataset_uri: str):
        """DLP scan before training"""
        parent = f"projects/{self.project_id}/locations/us"

        inspect_config = {
            "info_types": [
                {"name": "EMAIL_ADDRESS"},
                {"name": "PHONE_NUMBER"},
                {"name": "PERSON_NAME"}
            ],
            "min_likelihood": "POSSIBLE"
        }

        job = self.dlp_client.create_dlp_job(
            request={
                "parent": parent,
                "inspect_job": {
                    "storage_config": {
                        "cloud_storage_options": {
                            "file_set": {"url": f"{dataset_uri}/**"}
                        }
                    },
                    "inspect_config": inspect_config
                }
            }
        )

        print(f"DLP scan started: {job.name}")
        return job

    def create_secure_training_job(self, dataset_uri: str, checkpoint_uri: str):
        """Create VPC-SC protected training job"""
        job = aiplatform.CustomJob(
            display_name="arr-coc-secure-training",
            worker_pool_specs=[{
                "machine_spec": {
                    "machine_type": "a2-highgpu-8g",
                    "accelerator_type": "NVIDIA_TESLA_A100",
                    "accelerator_count": 8
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "us-west2-docker.pkg.dev/arr-coc-training/training/arr-coc:latest",
                    "args": [
                        f"--dataset={dataset_uri}",
                        f"--checkpoint-dir={checkpoint_uri}",
                        "--wandb-project=arr-coc-prod",
                        "--enable-audit-logging"
                    ]
                },
                # Security configuration
                "network": "projects/arr-coc-training/global/networks/arr-coc-vpc",
                "enable_web_access": False,
                "service_account": "training-job@arr-coc-training.iam.gserviceaccount.com"
            }],
            # CMEK encryption
            encryption_spec_key_name="projects/arr-coc-training/locations/us-west2/keyRings/ml-training/cryptoKeys/data-key",
            # Compliance labels
            labels={
                "compliance": "soc2",
                "environment": "production",
                "cost-center": "ml-research"
            }
        )

        return job

    def run_secure_pipeline(self, dataset_uri: str, checkpoint_uri: str):
        """Full secure training pipeline"""
        # Step 1: DLP scan
        dlp_job = self.scan_dataset(dataset_uri)

        # Step 2: Wait for DLP scan (in production, use async)
        # dlp_job.wait_until_complete()

        # Step 3: Create and run training job
        training_job = self.create_secure_training_job(dataset_uri, checkpoint_uri)
        training_job.run()

        return training_job

# Usage
trainer = SecureARRCOCTrainer(
    project_id="arr-coc-training",
    location="us-west2"
)

job = trainer.run_secure_pipeline(
    dataset_uri="gs://arr-coc-data/train",
    checkpoint_uri="gs://arr-coc-checkpoints/prod"
)
```

---

## 9. Monitoring and Troubleshooting

### 9.1 VPC-SC Dry Run Mode

```bash
# Test VPC-SC configuration without blocking traffic
gcloud access-context-manager perimeters update arr-coc-prod-perimeter \
    --set-enforced-mode=false \
    --set-dry-run-mode=true \
    --policy=POLICY_ID

# Monitor violations in dry-run mode
gcloud logging read '
    protoPayload.metadata.dryRun=true AND
    protoPayload.metadata.vpcServiceControlsUniqueId!=""
' \
    --limit=50 \
    --format=json
```

### 9.2 VPC-SC Violation Monitoring

```sql
-- BigQuery query for VPC-SC violations
SELECT
  timestamp,
  protoPayload.authenticationInfo.principalEmail,
  protoPayload.resourceName,
  protoPayload.serviceName,
  protoPayload.methodName,
  protoPayload.metadata.violationReason
FROM
  `PROJECT.audit_logs.cloudaudit_googleapis_com_data_access`
WHERE
  protoPayload.metadata.dryRun = FALSE
  AND protoPayload.metadata.vpcServiceControlsUniqueId IS NOT NULL
ORDER BY
  timestamp DESC
LIMIT 100
```

### 9.3 Network Connectivity Testing

```bash
# Test Private Google Access from VM
gcloud compute ssh test-vm \
    --zone=us-central1-a \
    --command="curl -I https://aiplatform.googleapis.com"

# Expected: 200 OK without public IP

# Test PSC endpoint
gcloud compute ssh test-vm \
    --zone=us-central1-a \
    --command="curl -I https://10.128.0.200"

# Expected: Vertex AI API response

# Verify DNS resolution
gcloud compute ssh test-vm \
    --zone=us-central1-a \
    --command="nslookup restricted.googleapis.com"

# Expected: 199.36.153.4
```

---

## Sources

**Google Cloud Documentation:**
- [VPC Service Controls Overview](https://docs.cloud.google.com/vpc-service-controls/docs/overview) (accessed 2025-11-16)
- [Vertex AI VPC Service Controls](https://docs.cloud.google.com/vertex-ai/docs/general/vpc-service-controls) (accessed 2025-11-16)
- [Ingress and Egress Rules](https://docs.cloud.google.com/vpc-service-controls/docs/ingress-egress-rules) (accessed 2025-11-16)
- [Private Google Access Configuration](https://cloud.google.com/vpc/docs/configure-private-google-access) (accessed 2025-11-16)
- [HIPAA Compliance on Google Cloud](https://docs.cloud.google.com/security/compliance/hipaa) (accessed 2025-11-16)
- [PCI-DSS Compliance](https://cloud.google.com/security/compliance/pci-dss) (accessed 2025-11-16)

**Tutorials and Codelabs:**
- [Private Service Connect Interface Vertex AI Pipelines](https://codelabs.developers.google.com/psc-interface-pipelines) (accessed 2025-11-16)

**Source Documents:**
- [gcp-vertex/00-custom-jobs-advanced.md](../gcp-vertex/00-custom-jobs-advanced.md) - VPC networking fundamentals, PSC vs VPC Peering comparison

**Web Research:**
- Google search: "VPC Service Controls Vertex AI perimeter 2024" (accessed 2025-11-16)
- Google search: "Private Google Access Vertex AI configuration" (accessed 2025-11-16)
- Google search: "Private Service Connect endpoints GCP Vertex AI" (accessed 2025-11-16)
- Google search: "data exfiltration prevention ML VPC Service Controls" (accessed 2025-11-16)
- Google search: "Shared VPC configuration Vertex AI host project service projects 2024" (accessed 2025-11-16)
- Google search: "VPC Service Controls ingress egress rules Vertex AI firewall" (accessed 2025-11-16)
- Google search: "Cloud DLP integration VPC Service Controls HIPAA PCI-DSS compliance" (accessed 2025-11-16)
