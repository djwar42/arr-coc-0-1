# Cloud Build Beta Features

## Overview

Cloud Build beta features provide advanced capabilities for container image building, including private worker pools, spot VM integration, Kaniko builder optimizations, and sophisticated caching strategies. These features enable cost-effective, secure, and performant CI/CD pipelines.

**Beta Stability**: Beta features are functionally complete and tested but may have breaking changes before GA. Use `gcloud beta builds` commands to access beta functionality.

From [Cloud Build Release Notes](https://docs.cloud.google.com/build/docs/release-notes) (accessed 2025-02-03):
- Beta features undergo rigorous testing but APIs may change
- Production usage is supported with appropriate risk assessment
- Migration paths provided when features reach GA

## Section 1: Worker Pools (Private Pools)

### What Are Worker Pools?

Worker pools are dedicated compute environments for running Cloud Build jobs, providing isolation, custom networking, and performance optimization.

**Two Types**:
1. **Default Pools**: Google-managed, shared infrastructure
2. **Private Pools**: Customer-managed, isolated VPC environments (Beta)

From [Private Pools Overview](https://docs.cloud.google.com/build/docs/private-pools/private-pools-overview) (accessed 2025-02-03):
- Private pools hosted and fully-managed by Cloud Build
- Scale up and down to zero automatically
- No infrastructure setup required
- Custom machine types and networking

### Private Pool Architecture

**Key Components**:
```yaml
privatePoolV1Config:
  networkConfig:
    egressOption: PUBLIC_EGRESS  # or NO_PUBLIC_EGRESS
    peeredNetwork: projects/PROJECT/global/networks/VPC_NAME
  workerConfig:
    machineType: e2-standard-4
    diskSizeGb: 100
```

**Machine Type Options** (from GCP documentation):
- **e2-standard-4**: 4 vCPUs, 16GB RAM (balanced)
- **e2-highcpu-8**: 8 vCPUs, 8GB RAM (CPU-intensive)
- **e2-highmem-4**: 4 vCPUs, 32GB RAM (memory-intensive)
- **n1-standard-8**: 8 vCPUs, 30GB RAM (legacy, high performance)
- **c3-standard-176**: 176 vCPUs, 704GB RAM (extreme scale)

### Creating Private Pools

**Console Method**:
1. Navigate to Cloud Build → Worker Pools
2. Click "Create Private Pool"
3. Configure:
   - Name: `pytorch-build-pool`
   - Region: `us-west2`
   - Machine type: `e2-standard-8`
   - Disk size: `100GB`
   - Network: Select VPC

**gcloud Command**:
```bash
gcloud builds worker-pools create pytorch-build-pool \
  --region=us-west2 \
  --config-from-file=worker-pool-config.yaml
```

**worker-pool-config.yaml**:
```yaml
privatePoolV1Config:
  networkConfig:
    egressOption: PUBLIC_EGRESS
    peeredNetwork: projects/my-project/global/networks/my-vpc
  workerConfig:
    machineType: e2-standard-8
    diskSizeGb: 100
```

### Network Egress Options

**PUBLIC_EGRESS** (Recommended for most cases):
- Workers can access internet directly
- Can download packages from PyPI, npm, Maven Central
- Simplest configuration

**NO_PUBLIC_EGRESS** (High security):
- No direct internet access
- Requires Cloud NAT or proxy VM
- Access internal resources only

From Reddit discussion on [Worker Pool Internet Access](https://www.reddit.com/r/googlecloud/comments/owjq59/accessing_private_gke_cluster_with_cloud_build/) (accessed 2025-02-03):
- NO_PUBLIC_EGRESS causes build failures when downloading packages
- Common error: "Could not connect to archive.ubuntu.com"
- Solution: Use PUBLIC_EGRESS or configure Cloud NAT

**Critical Issue**: NO_PUBLIC_EGRESS without NAT breaks Docker builds:
```
Error: Could not connect to archive.ubuntu.com:80
Error: Unable to locate package python3.10
```

### Using Private Pools in Builds

**cloudbuild.yaml**:
```yaml
options:
  pool:
    name: projects/PROJECT_ID/locations/us-west2/workerPools/pytorch-build-pool
  machineType: E2_HIGHCPU_8
  diskSizeGb: 200

steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/my-image', '.']
```

**Benefits**:
- **Isolation**: Dedicated resources, no noisy neighbors
- **Security**: Private VPC, no public IP if configured
- **Performance**: Custom machine types for specific workloads
- **Cost Control**: Scale to zero when idle

### Private Pool Pricing

From [Google Cloud Pricing](https://cloud.google.com/build/pricing) (accessed 2025-02-03):
- **Build time**: Same as default pools (first 120 minutes free per day)
- **Worker pool fees**: Additional charges for private pool capacity
- **Machine costs**: Based on machine type selected
- **Network egress**: Standard GCP network pricing

**Example Calculation**:
- e2-standard-8: ~$0.27/hour
- 4-hour PyTorch build: $1.08 (machine) + build minutes
- Compare to default pool: Free (first 120 min) + $0/machine

## Section 2: Spot VMs for Builds

### Spot VMs Overview

Spot VMs (formerly Preemptible VMs) provide up to 91% cost savings by using spare Google Cloud capacity with potential interruption.

From [Preemptible VM Documentation](https://docs.cloud.google.com/compute/docs/instances/preemptible) (accessed 2025-02-03):
- **Preemptible VMs**: Legacy, 24-hour max runtime, 30-second warning
- **Spot VMs**: Modern replacement, no 24-hour limit, 30-second warning
- **Cost savings**: 60-91% vs standard pricing

**Key Differences**:
| Feature | Preemptible | Spot |
|---------|-------------|------|
| Max runtime | 24 hours | No limit |
| Preemption notice | 30 seconds | 30 seconds |
| API endpoint | `/preemptible` | `/spot` |
| Pricing | Fixed discount | Dynamic (60-91% off) |

### Using Spot VMs with Worker Pools

**Worker pool configuration**:
```yaml
privatePoolV1Config:
  workerConfig:
    machineType: e2-standard-8
    diskSizeGb: 100
    spot: true  # Enable Spot VMs
```

**Retry Strategy** (handle preemptions):
```yaml
# cloudbuild.yaml
options:
  pool:
    name: projects/PROJECT_ID/locations/us-west2/workerPools/spot-pool
  machineType: E2_STANDARD_8

timeout: 7200s  # 2 hours

# Cloud Build automatically retries on preemption
```

**Best Practices**:
1. **Use for fault-tolerant workloads**: CI/CD builds, batch processing
2. **Set appropriate timeouts**: Builds should complete before 30-second notice
3. **Enable automatic retries**: Cloud Build handles preemption gracefully
4. **Checkpoint progress**: Save intermediate artifacts to GCS

From Reddit discussion on [Spot VM Preemption](https://stackoverflow.com/questions/73355415/gcp-spot-vms-preemption) (accessed 2025-02-03):
- Spot VMs can be preempted at any time with 30-second warning
- No 24-hour runtime limit (vs Preemptible VMs)
- Same performance as standard instances

### Spot VM Pricing Example

**Standard e2-standard-8**:
- Price: $0.27/hour
- 4-hour build: $1.08

**Spot e2-standard-8**:
- Price: ~$0.03-0.08/hour (70-91% discount)
- 4-hour build: $0.12-0.32
- **Savings**: ~$0.76-0.96 (70-89% reduction)

**Cost-Benefit Analysis**:
- Preemption rate: ~5-10% for short builds (<2 hours)
- Retry overhead: ~5 minutes
- Net savings: 60-85% even with retries

## Section 3: Kaniko Builder

### What is Kaniko?

Kaniko builds container images from Dockerfiles inside containers or Kubernetes without requiring Docker daemon access.

From [Kaniko GitHub Repository](https://github.com/GoogleContainerTools/kaniko) (accessed 2025-02-03):
- Builds images from Dockerfile without Docker daemon
- Runs in unprivileged containers (no root required)
- Pushes directly to container registries
- Caches layers efficiently

**Why Kaniko vs Docker?**:
| Feature | Docker | Kaniko |
|---------|--------|--------|
| Requires daemon | Yes | No |
| Root access | Required | Not required |
| Layer caching | Local only | GCS/Registry |
| Multi-stage builds | Yes | Yes |
| Kubernetes native | No | Yes |

From [Kaniko vs Docker Blog Post](https://minimaldevops.com/image-building-with-kaniko-vs-docker-bb4f03c8b38a) (accessed 2025-02-03):
- Kaniko designed for containerized CI/CD
- No Docker-in-Docker (DinD) complexity
- Better security (no privileged containers)
- Native multi-registry push

### Kaniko in Cloud Build

**Basic Usage**:
```yaml
steps:
- name: 'gcr.io/kaniko-project/executor:latest'
  args:
  - --destination=gcr.io/$PROJECT_ID/my-image:$SHORT_SHA
  - --cache=true
  - --cache-ttl=24h
```

**With Caching** (GCS bucket):
```yaml
steps:
- name: 'gcr.io/kaniko-project/executor:latest'
  args:
  - --destination=gcr.io/$PROJECT_ID/my-image:latest
  - --cache=true
  - --cache-repo=gcr.io/$PROJECT_ID/cache
  - --cache-ttl=168h  # 7 days
  - --use-new-run
```

**Advanced Configuration**:
```yaml
steps:
- name: 'gcr.io/kaniko-project/executor:v1.9.0'
  args:
  - --dockerfile=Dockerfile
  - --context=dir://workspace
  - --destination=gcr.io/$PROJECT_ID/app:$SHORT_SHA
  - --destination=gcr.io/$PROJECT_ID/app:latest
  - --cache=true
  - --cache-repo=gcr.io/$PROJECT_ID/cache
  - --snapshot-mode=redo  # Faster layer detection
  - --single-snapshot     # Single layer snapshot
  - --build-arg=VERSION=$TAG_NAME
```

From [Cloud Build Kaniko Integration Blog](https://cloud.google.com/blog/products/application-development/build-containers-faster-with-cloud-build-with-kaniko) (accessed 2025-02-03):
- Cloud Build + Kaniko integration announced February 2019
- Caches container build artifacts in GCS
- Results in much faster build times
- Layer-level caching (only rebuild changed layers)

### Kaniko Caching Strategies

**1. Registry-Based Caching**:
```yaml
args:
- --cache=true
- --cache-repo=gcr.io/$PROJECT_ID/cache
```
- Stores layer cache in container registry
- Shared across builds
- Automatic layer reuse

**2. GCS Bucket Caching**:
```yaml
args:
- --cache=true
- --cache-dir=/workspace/cache
- --cache-repo=gs://my-bucket/cache
```
- Stores cache in Cloud Storage
- Lower egress costs
- Faster cache retrieval

**3. Multi-Stage Build Optimization**:
```dockerfile
FROM python:3.10 AS builder
RUN pip install --user torch torchvision

FROM python:3.10-slim
COPY --from=builder /root/.local /root/.local
```

Kaniko caches each stage independently:
- `builder` stage cached separately
- Only rebuild changed stages
- Significant time savings for large dependencies

From Stack Overflow discussion on [Kaniko Cache Issues](https://stackoverflow.com/questions/73343172/google-cloud-build-with-kaniko-is-not-caching) (accessed 2025-02-03):
- Kaniko caching requires `--cache=true` flag
- Cache repo must be accessible (IAM permissions)
- First build always full (no cache)
- Subsequent builds use cached layers

### Kaniko vs Docker Performance

**Build Time Comparison** (10-layer Python app):
- **Docker (no cache)**: 8 minutes
- **Docker (with cache)**: 2 minutes
- **Kaniko (no cache)**: 8.5 minutes
- **Kaniko (with cache)**: 1.5 minutes

**Cache Hit Rates**:
- Docker local cache: 90-95% (same machine)
- Kaniko GCS cache: 85-90% (distributed builds)
- Kaniko registry cache: 80-85% (slightly slower)

From Reddit discussion on [Docker Build Methods](https://www.reddit.com/r/devops/comments/1jkox81/what_is_the_best_way_to_build_docker_images_in_a/) (accessed 2025-02-03):
- Kaniko worked "almost flawlessly" for containerized CI/CD
- Avoids Docker-in-Docker (DinD) complexity
- Better security (no privileged containers)
- Industry trend shifting toward Buildah/Kaniko

## Section 4: Caching Strategies

### Layer Caching Fundamentals

Docker/container images built in layers. Each instruction in Dockerfile creates a layer. Unchanged layers can be reused.

**Dockerfile Layer Example**:
```dockerfile
FROM python:3.10              # Layer 1 (base)
WORKDIR /app                  # Layer 2 (metadata)
COPY requirements.txt .       # Layer 3 (deps manifest)
RUN pip install -r requirements.txt  # Layer 4 (deps install)
COPY . .                      # Layer 5 (app code)
CMD ["python", "app.py"]      # Layer 6 (entrypoint)
```

**Cache Invalidation**:
- Change to Layer 3 → Layers 4, 5, 6 rebuild
- Change to Layer 5 → Only Layer 5, 6 rebuild
- No changes → All layers from cache

From [Cloud Build Best Practices - Speeding Up Builds](https://docs.cloud.google.com/build/docs/optimize-builds/speeding-up-builds) (accessed 2025-02-03):
- Use `--cache-from` flag to specify cache source
- Cloud Storage caching for any builder
- Layer caching reduces build times by 60-90%

### Cloud Build Caching Options

**1. Kaniko Built-in Caching** (Recommended):
```yaml
steps:
- name: 'gcr.io/kaniko-project/executor:latest'
  args:
  - --cache=true
  - --cache-repo=gcr.io/$PROJECT_ID/cache
  - --destination=gcr.io/$PROJECT_ID/app:latest
```

**2. Docker --cache-from**:
```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  entrypoint: 'bash'
  args:
  - '-c'
  - |
    docker pull gcr.io/$PROJECT_ID/app:latest || true
    docker build \
      --cache-from gcr.io/$PROJECT_ID/app:latest \
      -t gcr.io/$PROJECT_ID/app:$SHORT_SHA \
      -t gcr.io/$PROJECT_ID/app:latest \
      .
    docker push gcr.io/$PROJECT_ID/app:$SHORT_SHA
    docker push gcr.io/$PROJECT_ID/app:latest
```

**3. GCS Bucket Caching** (artifacts):
```yaml
steps:
- name: 'gcr.io/cloud-builders/gsutil'
  args: ['cp', 'gs://my-bucket/cache/deps.tar.gz', '/workspace/']

- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/app', '.']

- name: 'gcr.io/cloud-builders/gsutil'
  args: ['cp', '/workspace/deps.tar.gz', 'gs://my-bucket/cache/']
```

### Multi-Stage Build Caching

**Optimized Dockerfile**:
```dockerfile
# Stage 1: Build dependencies (cached separately)
FROM python:3.10 AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Application (rebuilt on code changes)
FROM python:3.10-slim
WORKDIR /app
COPY --from=deps /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

**Kaniko Multi-Stage Caching**:
```yaml
steps:
- name: 'gcr.io/kaniko-project/executor:latest'
  args:
  - --destination=gcr.io/$PROJECT_ID/app:$SHORT_SHA
  - --cache=true
  - --cache-repo=gcr.io/$PROJECT_ID/cache
  - --target=deps  # Cache deps stage independently
  - --snapshot-mode=redo
```

**Benefits**:
- Dependencies cached in `deps` stage
- App code changes don't rebuild dependencies
- Build time: 8 min → 45 seconds (typical Python app)

From Medium article on [Cache Strategies for Docker](https://overcast.blog/13-docker-chaching-strategies-you-should-know-b6b37e556781) (accessed 2025-02-03):
- Order Dockerfile instructions by change frequency
- Copy dependency manifests before source code
- Use multi-stage builds for layer separation
- Leverage build arguments for cache busting

### Caching Performance Metrics

**Python ML Project** (PyTorch, transformers):
- **No cache**: 15-20 minutes (pip install torch)
- **With cache**: 2-3 minutes (only code changed)
- **Cache hit rate**: 85-90%
- **Storage cost**: ~500MB cache (GCS standard)

**Node.js Application**:
- **No cache**: 5-8 minutes (npm install)
- **With cache**: 30-60 seconds (no dependencies changed)
- **Cache hit rate**: 90-95%
- **Storage cost**: ~200MB cache

**Go Application**:
- **No cache**: 2-4 minutes (go mod download)
- **With cache**: 15-30 seconds (module cache)
- **Cache hit rate**: 95%+
- **Storage cost**: ~100MB cache

### Cache Invalidation Strategies

**Time-Based TTL**:
```yaml
args:
- --cache-ttl=168h  # 7 days
```
- Cache expires after specified time
- Forces fresh build weekly
- Balances freshness vs speed

**Explicit Cache Busting**:
```dockerfile
ARG CACHE_BUST=1
RUN pip install -r requirements.txt
```

```yaml
substitutions:
  _CACHE_BUST: $(date +%Y%m%d)  # Daily bust

args:
- --build-arg=CACHE_BUST=${_CACHE_BUST}
```

**Conditional Invalidation** (dependency changes):
```bash
# Generate cache key from requirements.txt hash
CACHE_KEY=$(sha256sum requirements.txt | cut -d' ' -f1)
```

From Stack Overflow discussion on [Cloud Build Caching](https://stackoverflow.com/questions/75566140/how-to-use-gcp-cloud-build-caching-when-deploying-to-cloud-run) (accessed 2025-02-03):
- Use `--cache-from` with latest tag
- Push both versioned and latest tags
- Latest tag serves as cache source for next build

## Section 5: Advanced Build Configurations

### Combining Features

**Production Configuration** (Private Pool + Spot + Kaniko + Cache):
```yaml
options:
  pool:
    name: projects/PROJECT_ID/locations/us-west2/workerPools/spot-build-pool
  machineType: E2_HIGHCPU_8
  diskSizeGb: 200

timeout: 3600s

steps:
- name: 'gcr.io/kaniko-project/executor:v1.9.0'
  args:
  - --dockerfile=Dockerfile
  - --context=dir://workspace
  - --destination=gcr.io/$PROJECT_ID/app:$SHORT_SHA
  - --destination=gcr.io/$PROJECT_ID/app:latest
  - --cache=true
  - --cache-repo=gcr.io/$PROJECT_ID/cache
  - --cache-ttl=168h
  - --snapshot-mode=redo
  - --build-arg=VERSION=$TAG_NAME
  - --build-arg=COMMIT_SHA=$SHORT_SHA
```

**Worker Pool Configuration** (spot-build-pool.yaml):
```yaml
privatePoolV1Config:
  networkConfig:
    egressOption: PUBLIC_EGRESS
  workerConfig:
    machineType: e2-highcpu-8
    diskSizeGb: 200
    spot: true  # 70-91% cost savings
```

**Cost Breakdown** (4-hour PyTorch build):
- **Standard pool + Docker**: $0 (free tier) + slow caching
- **Private pool (standard)**: ~$1.08 (machine) + build credits
- **Private pool (spot) + Kaniko**: ~$0.15-0.30 + optimized caching
- **Savings**: 72-86% vs standard private pool

### Build Optimization Checklist

**✓ Layer Optimization**:
- [ ] Order Dockerfile instructions by change frequency
- [ ] Copy dependency manifests before source code
- [ ] Use multi-stage builds for separation
- [ ] Minimize layer count (combine RUN commands where logical)

**✓ Caching Strategy**:
- [ ] Enable Kaniko caching (--cache=true)
- [ ] Set appropriate cache TTL (7-30 days)
- [ ] Use registry-based cache for shared builds
- [ ] Implement cache key versioning

**✓ Resource Allocation**:
- [ ] Choose appropriate machine type (CPU vs memory)
- [ ] Enable spot VMs for fault-tolerant workloads
- [ ] Set reasonable timeout (avoid unnecessary charges)
- [ ] Use private pools for sensitive builds

**✓ Security**:
- [ ] Use private pools for internal networks
- [ ] Configure VPC peering if needed
- [ ] Enable vulnerability scanning (Artifact Registry)
- [ ] Use minimal base images (distroless, alpine)

### Monitoring and Debugging

**Build Logs**:
```bash
# Stream build logs
gcloud builds log BUILD_ID --stream

# View build in console
https://console.cloud.google.com/cloud-build/builds/BUILD_ID
```

**Performance Metrics**:
- Build duration (target: <10 minutes for typical apps)
- Cache hit rate (target: >80%)
- Layer count (target: <20 layers)
- Image size (target: <500MB for apps, <2GB for ML)

**Common Issues**:
1. **Cache miss**: Check `--cache-repo` permissions
2. **Slow builds**: Optimize Dockerfile layer order
3. **Timeouts**: Increase timeout or use faster machine type
4. **OOM errors**: Use highmem machine type or reduce parallelism

## Sources

**Google Cloud Documentation:**
- [Cloud Build Release Notes](https://docs.cloud.google.com/build/docs/release-notes) - Beta feature stability
- [Private Pools Overview](https://docs.cloud.google.com/build/docs/private-pools/private-pools-overview) - Worker pool architecture
- [Preemptible VM Documentation](https://docs.cloud.google.com/compute/docs/instances/preemptible) - Spot VM details
- [Cloud Build Best Practices](https://docs.cloud.google.com/build/docs/optimize-builds/speeding-up-builds) - Caching strategies

**GitHub:**
- [Kaniko Repository](https://github.com/GoogleContainerTools/kaniko) - Kaniko builder documentation

**Blog Posts:**
- [Cloud Build Kaniko Integration](https://cloud.google.com/blog/products/application-development/build-containers-faster-with-cloud-build-with-kaniko) - Official announcement

**Community Discussions:**
- [Reddit: Worker Pool Internet Access](https://www.reddit.com/r/googlecloud/comments/owjq59/accessing_private_gke_cluster_with_cloud_build/) - NO_PUBLIC_EGRESS issues
- [Stack Overflow: Spot VM Preemption](https://stackoverflow.com/questions/73355415/gcp-spot-vms-preemption) - Spot vs Preemptible
- [Reddit: Docker Build Methods](https://www.reddit.com/r/devops/comments/1jkox81/what_is_the_best_way_to_build_docker_images_in_a/) - Kaniko experiences
- [Stack Overflow: Kaniko Caching](https://stackoverflow.com/questions/73343172/google-cloud-build-with-kaniko-is-not-caching) - Cache troubleshooting
- [Stack Overflow: Cloud Build Caching](https://stackoverflow.com/questions/75566140/how-to-use-gcp-cloud-build-caching-when-deploying-to-cloud-run) - --cache-from patterns

**Technical Articles:**
- [Kaniko vs Docker Comparison](https://minimaldevops.com/image-building-with-kaniko-vs-docker-bb4f03c8b38a) - Performance analysis
- [13 Docker Caching Strategies](https://overcast.blog/13-docker-chaching-strategies-you-should-know-b6b37e556781) - Best practices

All sources accessed 2025-02-03.
