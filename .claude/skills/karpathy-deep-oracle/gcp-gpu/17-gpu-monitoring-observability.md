# GPU Monitoring & Observability

Comprehensive guide to monitoring GPU workloads on Google Cloud Platform with DCGM, Cloud Monitoring, Prometheus, Grafana, and alerting strategies.

---

## Overview

GPU monitoring is critical for:
- **Cost control** - Identify idle/underutilized GPUs ($2-15/hour waste)
- **Performance optimization** - Detect bottlenecks, memory leaks, thermal throttling
- **Reliability** - Prevent OOM crashes, overheating failures
- **Capacity planning** - Track utilization trends for quota/scaling decisions

**Key Metrics**:
- Utilization (compute %, memory %)
- Temperature (°C)
- Power consumption (W)
- Memory allocation (GB used/total)
- SM (Streaming Multiprocessor) occupancy
- PCIe throughput (GB/s)

---

## Section 1: Cloud Monitoring GPU Metrics

**Native GCP Integration** - Cloud Monitoring automatically collects GPU metrics from Compute Engine instances with GPUs attached.

### Built-in Metrics

From [Google Cloud Monitoring release notes](https://docs.cloud.google.com/stackdriver/docs/release-notes) (accessed 2025-11-16):

**Compute Engine GPU metrics** (automatically available):
```
compute.googleapis.com/instance/gpu/utilization       # GPU compute utilization (0-1)
compute.googleapis.com/instance/gpu/memory_utilization # GPU memory utilization (0-1)
compute.googleapis.com/instance/gpu/temperature        # GPU temperature (°C)
```

**Access via Cloud Console**:
1. Navigate to Monitoring → Metrics Explorer
2. Select resource type: "VM Instance"
3. Filter by GPU metrics prefix: `compute.googleapis.com/instance/gpu/`
4. Visualize per-GPU or aggregate across instances

**Querying with MQL (Monitoring Query Language)**:
```sql
fetch gce_instance
| metric 'compute.googleapis.com/instance/gpu/utilization'
| group_by [resource.instance_id, metric.gpu_number]
| every 1m
| mean()
```

**Limitations**:
- Basic metrics only (no per-process breakdown)
- 1-minute granularity (not real-time)
- No GPU memory allocation details (only % utilization)
- No kernel-level profiling data

**Cost**: Cloud Monitoring GPU metrics included in standard GCP pricing (no extra charge for ingestion, but alerting policies incur costs starting January 7, 2025).

---

## Section 2: DCGM (Data Center GPU Manager) Integration

**NVIDIA DCGM** - Comprehensive GPU management and monitoring suite for datacenter deployments.

From [Google Cloud Ops Agent DCGM integration](https://docs.cloud.google.com/monitoring/agent/ops-agent/third-party-nvidia) (accessed 2025-11-16) and [DCGM Developer documentation](https://developer.nvidia.com/dcgm):

### DCGM Architecture

**Components**:
- **DCGM Daemon** (`nv-hostengine`) - Runs on each GPU node, collects metrics
- **DCGM Exporter** - Exports metrics in Prometheus format
- **Cloud Monitoring Integration** - Ops Agent can collect DCGM metrics and send to Cloud Monitoring

**Advanced Metrics** (beyond Cloud Monitoring built-ins):
```
# Performance
DCGM_FI_PROF_GR_ENGINE_ACTIVE          # Graphics engine active time (%)
DCGM_FI_PROF_SM_ACTIVE                 # Streaming multiprocessor active (%)
DCGM_FI_PROF_SM_OCCUPANCY              # SM occupancy (%)
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE        # Tensor core active (%)
DCGM_FI_PROF_DRAM_ACTIVE               # DRAM active (%)
DCGM_FI_PROF_PCIE_TX_BYTES             # PCIe TX throughput (bytes)
DCGM_FI_PROF_PCIE_RX_BYTES             # PCIe RX throughput (bytes)

# Health
DCGM_FI_DEV_XID_ERRORS                 # XID error count
DCGM_FI_DEV_POWER_VIOLATION            # Power limit violation time (μs)
DCGM_FI_DEV_THERMAL_VIOLATION          # Thermal limit violation time (μs)
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL          # ECC single-bit errors
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL          # ECC double-bit errors
```

### Installation (Compute Engine)

**Option 1: Ops Agent + Cloud Monitoring** (recommended for GCP)

From [Ops Agent DCGM third-party integration](https://docs.cloud.google.com/monitoring/agent/ops-agent/third-party-nvidia):

```bash
# 1. Install NVIDIA driver and CUDA (if not already installed)
sudo /opt/deeplearning/install-driver.sh  # On Deep Learning VM images
# OR manually:
# curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update && sudo apt install -y cuda-drivers

# 2. Install DCGM (version 3.1-3.3.9 compatible with Ops Agent)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y datacenter-gpu-manager

# 3. Start DCGM daemon
sudo systemctl --now enable nvidia-dcgm

# 4. Install Google Cloud Ops Agent
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install

# 5. Configure Ops Agent to collect DCGM metrics
sudo tee /etc/google-cloud-ops-agent/config.yaml <<EOF
metrics:
  receivers:
    dcgm:
      type: dcgm
      collection_interval: 30s
  service:
    pipelines:
      dcgm:
        receivers: [dcgm]
EOF

# 6. Restart Ops Agent
sudo service google-cloud-ops-agent restart

# 7. Verify metrics collection
dcgmi discovery -l           # List GPUs detected by DCGM
dcgmi dmon -e 150,155,156    # Monitor utilization, memory, temp
```

**DCGM Dashboard Auto-Installation**: Once DCGM metrics begin flowing to Cloud Monitoring, a pre-built dashboard is automatically added to your GCP project (no manual creation needed).

**Option 2: Self-Managed DCGM Exporter + Prometheus**

From [Medium: Self-managed GPU Monitoring Stack on Google Cloud with DCGM, Prometheus, and Grafana](https://medium.com/google-cloud/self-managed-gpu-monitoring-stack-on-google-cloud-with-dcgm-prometheus-and-grafana-04c8355c5132) (accessed 2025-11-16):

```bash
# 1. Install DCGM (as above)

# 2. Install DCGM Exporter (Prometheus format)
docker run -d --rm \
  --gpus all \
  --net host \
  --cap-add SYS_ADMIN \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.0-ubuntu22.04

# 3. Verify exporter running
curl localhost:9400/metrics | grep DCGM

# Output example:
# DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-abc123..."} 87
# DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",...} 45
# DCGM_FI_DEV_GPU_TEMP{gpu="0",...} 72
```

**DCGM Exporter for Kubernetes** (GKE):
```yaml
# DaemonSet ensures DCGM exporter runs on all GPU nodes
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: gpu-monitoring
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: "nvidia-tesla-a100"  # Target GPU nodes
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.0-ubuntu22.04
        ports:
        - containerPort: 9400
          name: metrics
        env:
        - name: DCGM_EXPORTER_LISTEN
          value: ":9400"
        - name: DCGM_EXPORTER_KUBERNETES
          value: "true"
        securityContext:
          capabilities:
            add: ["SYS_ADMIN"]
        volumeMounts:
        - name: pod-resources
          mountPath: /var/lib/kubelet/pod-resources
      volumes:
      - name: pod-resources
        hostPath:
          path: /var/lib/kubelet/pod-resources
---
# Service for Prometheus scraping
apiVersion: v1
kind: Service
metadata:
  name: dcgm-exporter
  namespace: gpu-monitoring
  labels:
    app: dcgm-exporter
spec:
  selector:
    app: dcgm-exporter
  ports:
  - port: 9400
    name: metrics
```

---

## Section 3: nvidia-smi Automation & Monitoring

**nvidia-smi** - NVIDIA System Management Interface, CLI tool for querying GPU state.

From [NVIDIA DCGM documentation](https://developer.nvidia.com/dcgm) and community best practices:

### nvidia-smi Commands

**Basic monitoring**:
```bash
# Real-time GPU stats (refreshes every 1 second)
nvidia-smi dmon -s pucvmet -d 1
# p: power, u: utilization, c: proc count, v: violations, m: memory, e: ecc, t: temp

# Query specific fields in CSV format
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1

# Per-process GPU memory usage
nvidia-smi pmon -s m -d 1
# Shows which processes are consuming GPU memory
```

**Logging to file**:
```bash
# Continuous logging for post-mortem analysis
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv -l 10 >> /var/log/gpu-stats.csv &

# Structured logging (JSON-compatible)
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader -l 5 | while read line; do
  echo "{\"timestamp\":\"$(date -Iseconds)\",\"metrics\":\"$line\"}" >> /var/log/gpu-metrics.json
done
```

### Prometheus Exporter

From [GitHub: NVIDIA/dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) and [community nvidia-smi exporters](https://github.com/utkuozdemir/nvidia_gpu_exporter):

**nvidia-smi Prometheus exporter** (alternative to DCGM for simpler setups):

```bash
# Using nvidia_gpu_exporter (lightweight, nvidia-smi based)
docker run -d --rm \
  --gpus all \
  -p 9835:9835 \
  utkuozdemir/nvidia_gpu_exporter:1.2.0

# Metrics endpoint
curl localhost:9835/metrics

# Output includes:
# nvidia_gpu_duty_cycle{minor="0",name="NVIDIA A100-SXM4-80GB",uuid="..."} 0.87
# nvidia_gpu_memory_used_bytes{minor="0",...} 68719476736
# nvidia_gpu_temperature_celsius{minor="0",...} 72
# nvidia_gpu_power_usage_milliwatts{minor="0",...} 350000
```

**Prometheus scrape configuration**:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'nvidia-gpu-exporter'
    static_configs:
      - targets: ['gpu-vm-1:9835', 'gpu-vm-2:9835']
    scrape_interval: 30s
    metrics_path: /metrics
```

---

## Section 4: Prometheus + Grafana GPU Dashboards

**Prometheus** - Time-series database for metrics storage and querying.
**Grafana** - Visualization platform for creating dashboards.

From [Grafana GPU monitoring dashboards](https://grafana.com/grafana/dashboards/9822-gpu-monitoring/) and [Grafana AI Observability](https://grafana.com/docs/grafana-cloud/monitor-applications/ai-observability/gpu-observability/setup/) (accessed 2025-11-16):

### Prometheus Setup (GCE)

```bash
# Install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz
tar -xvf prometheus-2.54.1.linux-amd64.tar.gz
cd prometheus-2.54.1.linux-amd64

# Configure Prometheus to scrape DCGM exporter
cat > prometheus.yml <<EOF
global:
  scrape_interval: 30s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'dcgm-exporter'
    static_configs:
      - targets:
        - 'gpu-node-1:9400'
        - 'gpu-node-2:9400'
        labels:
          cluster: 'training-cluster'
          region: 'us-central1'

  - job_name: 'nvidia-gpu-exporter'
    static_configs:
      - targets: ['inference-node-1:9835']
    scrape_interval: 15s  # Higher frequency for inference monitoring
EOF

# Start Prometheus
./prometheus --config.file=prometheus.yml --storage.tsdb.retention.time=30d
```

### Grafana Dashboard Setup

From [Grafana GPU monitoring](https://grafana.com/grafana/dashboards/9822-gpu-monitoring/) and [LeaderGPU Grafana guide](https://www.leadergpu.com/articles/524-collecting-gpu-metrics-with-grafana):

**Install Grafana**:
```bash
# Add Grafana APT repository
sudo apt-get install -y apt-transport-https software-properties-common
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list

# Install Grafana
sudo apt update && sudo apt install -y grafana

# Start Grafana
sudo systemctl enable --now grafana-server

# Access at http://localhost:3000 (default user/pass: admin/admin)
```

**Configure Prometheus data source** (Grafana UI):
1. Configuration → Data Sources → Add data source → Prometheus
2. URL: `http://localhost:9090`
3. Save & Test

**Import pre-built GPU dashboard**:
1. Dashboards → Import → Dashboard ID: `9822` (NVIDIA GPU Kubernetes Monitoring)
2. OR create custom dashboard with panels:

**Example panel queries (PromQL)**:

**GPU Utilization**:
```promql
# Average GPU utilization across all GPUs
avg(DCGM_FI_DEV_GPU_UTIL) by (gpu, instance)

# Per-GPU utilization with threshold highlighting
DCGM_FI_DEV_GPU_UTIL > 80  # Highlight high utilization
```

**GPU Memory Usage**:
```promql
# Memory used (GB)
DCGM_FI_DEV_FB_USED / 1024

# Memory utilization percentage
(DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE) * 100
```

**GPU Temperature**:
```promql
# Current temperature
DCGM_FI_DEV_GPU_TEMP

# Temperature trend (5-minute moving average)
avg_over_time(DCGM_FI_DEV_GPU_TEMP[5m])
```

**Power Consumption**:
```promql
# Current power draw (W)
DCGM_FI_DEV_POWER_USAGE

# Power efficiency (FLOPS per watt - estimated)
(DCGM_FI_DEV_GPU_UTIL * 312) / DCGM_FI_DEV_POWER_USAGE  # 312 TFLOPS for A100
```

**GPU Idle Detection** (cost waste):
```promql
# GPUs with <5% utilization for last 10 minutes
avg_over_time(DCGM_FI_DEV_GPU_UTIL[10m]) < 5
```

**Dashboard Features** (from Grafana AI Observability):
- **Hardware monitoring** - GPU utilization, temperature, power consumption
- **Memory tracking** - GPU memory usage and allocation patterns
- **Multi-GPU visualization** - Side-by-side comparison of GPU performance
- **Alerts integration** - Visual indicators for threshold violations

---

## Section 5: Alerting Policies

**Proactive monitoring** - Detect issues before they cause failures or cost overruns.

From Google Cloud Monitoring best practices:

### Cloud Monitoring Alerting

**Create alert policy** (gcloud CLI):
```bash
# Alert on low GPU utilization (cost waste detection)
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="GPU Underutilization Alert" \
  --condition-display-name="GPU utilization < 10% for 30 minutes" \
  --condition-threshold-value=0.10 \
  --condition-threshold-duration=1800s \
  --condition-threshold-comparison=COMPARISON_LT \
  --condition-threshold-aggregations=mean,ALIGN_MEAN,1800s \
  --condition-threshold-resource-type=gce_instance \
  --condition-threshold-metric-filter='metric.type="compute.googleapis.com/instance/gpu/utilization"'

# Alert on high GPU temperature (thermal throttling risk)
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="GPU Overheating Alert" \
  --condition-display-name="GPU temperature > 85°C" \
  --condition-threshold-value=85 \
  --condition-threshold-duration=300s \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-aggregations=max,ALIGN_MAX,300s \
  --condition-threshold-resource-type=gce_instance \
  --condition-threshold-metric-filter='metric.type="compute.googleapis.com/instance/gpu/temperature"'

# Alert on GPU OOM (out of memory)
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="GPU Memory Exhaustion" \
  --condition-display-name="GPU memory utilization > 95%" \
  --condition-threshold-value=0.95 \
  --condition-threshold-duration=60s \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-aggregations=mean,ALIGN_MEAN,60s \
  --condition-threshold-resource-type=gce_instance \
  --condition-threshold-metric-filter='metric.type="compute.googleapis.com/instance/gpu/memory_utilization"'
```

**Notification channels**:
```bash
# Create email notification channel
gcloud alpha monitoring channels create \
  --display-name="GPU Ops Team" \
  --type=email \
  --channel-labels=email_address=gpu-ops@example.com

# Create Slack notification channel
gcloud alpha monitoring channels create \
  --display-name="Slack #gpu-alerts" \
  --type=slack \
  --channel-labels=url=https://hooks.slack.com/services/T00/B00/XXX
```

### Prometheus Alertmanager

From [NVIDIA GPU Telemetry documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/kube-prometheus.html) and Prometheus best practices:

**Alert rules** (alert.rules.yml):
```yaml
groups:
- name: gpu_alerts
  interval: 30s
  rules:
  # Low utilization (cost waste)
  - alert: GPUIdleForTooLong
    expr: avg_over_time(DCGM_FI_DEV_GPU_UTIL[30m]) < 5
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "GPU {{ $labels.gpu }} on {{ $labels.instance }} idle for 30+ minutes"
      description: "GPU utilization: {{ $value }}%. Consider terminating instance to save costs ($2-15/hour)."

  # High temperature (thermal throttling)
  - alert: GPUOverheating
    expr: DCGM_FI_DEV_GPU_TEMP > 85
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "GPU {{ $labels.gpu }} overheating on {{ $labels.instance }}"
      description: "Temperature: {{ $value }}°C. Risk of thermal throttling or hardware damage."

  # Memory exhaustion
  - alert: GPUMemoryExhaustion
    expr: (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE) > 0.95
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "GPU {{ $labels.gpu }} memory exhausted on {{ $labels.instance }}"
      description: "Memory usage: {{ $value | humanizePercentage }}. OOM crash imminent."

  # ECC errors (hardware degradation)
  - alert: GPUECCErrors
    expr: increase(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL[1h]) > 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "GPU {{ $labels.gpu }} ECC errors detected on {{ $labels.instance }}"
      description: "Double-bit ECC errors: {{ $value }}. Hardware failure likely."

  # Power limit violations
  - alert: GPUPowerThrottling
    expr: increase(DCGM_FI_DEV_POWER_VIOLATION[10m]) > 60000000  # 60 seconds in μs
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "GPU {{ $labels.gpu }} power throttling on {{ $labels.instance }}"
      description: "Power limit violations for {{ $value | humanizeDuration }}. Performance degraded."

  # Batch job completion rate (training-specific)
  - alert: TrainingJobStalled
    expr: rate(training_batch_completed_total[5m]) == 0
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Training job stalled on {{ $labels.instance }}"
      description: "No batch completions in 15 minutes. GPU may be hung or job crashed."
```

**Alertmanager configuration** (alertmanager.yml):
```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'gpu-ops-team'
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty-critical'
    continue: true
  - match:
      severity: warning
    receiver: 'slack-warnings'

receivers:
- name: 'gpu-ops-team'
  email_configs:
  - to: 'gpu-ops@example.com'
    from: 'alertmanager@example.com'
    smarthost: smtp.gmail.com:587
    auth_username: 'alerts@example.com'
    auth_password: 'app-password'

- name: 'slack-warnings'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/T00/B00/XXX'
    channel: '#gpu-alerts'
    title: 'GPU Warning: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_KEY'
    description: '{{ .CommonAnnotations.summary }}'
```

---

## Section 6: GPU Memory Leak Detection

**Memory leaks** - Gradual increase in GPU memory usage without corresponding deallocation, leading to OOM crashes.

From [PyTorch memory leak debugging](https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741) and [NVIDIA Compute Sanitizer](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/):

### Detection Patterns

**Symptom 1: Gradual memory growth**
```promql
# Prometheus query: Detect linear memory growth
deriv(DCGM_FI_DEV_FB_USED[1h]) > 0  # Positive slope over 1 hour
```

**Symptom 2: Memory not released between epochs**
```bash
# nvidia-smi before epoch
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# nvidia-smi after epoch (should return to baseline, but doesn't)
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### Debugging Tools

**NVIDIA Compute Sanitizer** (detect CUDA memory leaks):

From [NVIDIA Compute Sanitizer blog post](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/) (June 29, 2023):

```bash
# Run training script with memory leak detection
compute-sanitizer --leak-check full python train.py

# Output example:
# ========= LEAK SUMMARY
# ========= cudaMalloc 8589934592 bytes (leaked at main.cu:42)
# ========= Total leaked memory: 8 GB
```

**PyTorch memory profiler**:
```python
import torch
import gc

# Before training loop
torch.cuda.reset_peak_memory_stats()
baseline_allocated = torch.cuda.memory_allocated()
baseline_reserved = torch.cuda.memory_reserved()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # After each epoch, check for leaks
    current_allocated = torch.cuda.memory_allocated()
    current_reserved = torch.cuda.memory_reserved()

    leak_allocated = current_allocated - baseline_allocated
    leak_reserved = current_reserved - baseline_reserved

    if leak_allocated > 1e9:  # 1 GB leak threshold
        print(f"Epoch {epoch}: Memory leak detected! {leak_allocated / 1e9:.2f} GB allocated but not freed")

        # Inspect live tensors
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(f"Tensor: {obj.size()}, {obj.dtype}, {obj.device}")

# Common leak causes (PyTorch):
# 1. Holding references to loss/outputs outside training loop
# 2. Accumulating gradients without .detach()
# 3. Circular references in custom modules
# 4. Not calling optimizer.zero_grad()
```

**Memory leak checklist**:
- [ ] All `.backward()` calls followed by `optimizer.zero_grad()`
- [ ] No global variables accumulating tensors
- [ ] `.detach()` called on tensors used for logging/metrics
- [ ] DataLoader workers properly cleaned up
- [ ] CUDA streams explicitly synchronized and freed
- [ ] Custom CUDA extensions properly deallocate memory

---

## Section 7: Cost Monitoring Per GPU

**Cost attribution** - Track spending per GPU, per project, per team for chargeback and budgeting.

From [GCP committed use discounts](https://docs.cloud.google.com/compute/docs/instances/committed-use-discounts-overview) and [GPU pricing](https://cloud.google.com/compute/gpus-pricing) (accessed 2025-11-16):

### GPU Pricing Structure (2024-2025)

**On-demand pricing** (us-central1, hourly):
```
NVIDIA A100 80GB: $3.673/hour
NVIDIA H100 80GB: $5.505/hour
NVIDIA L4: $0.888/hour
NVIDIA T4: $0.35/hour
NVIDIA V100: $2.48/hour
```

**Committed Use Discounts (CUDs)** - Up to 57% savings for 1-year or 3-year commitments:
```
1-year CUD: 37% discount
3-year CUD: 57% discount

Example (A100 80GB):
On-demand: $3.673/hour = $2,656/month
1-year CUD: $2.31/hour = $1,671/month (37% savings = $985/month)
3-year CUD: $1.58/hour = $1,142/month (57% savings = $1,514/month)
```

**Sustained Use Discounts (SUDs)** - Automatic discounts for sustained usage (now deprecated for GPUs, replaced by CUDs).

**Spot/Preemptible GPU pricing** - 60-91% discount (but can be preempted):
```
Spot A100 80GB: $1.10/hour (70% discount)
Spot H100 80GB: $1.65/hour (70% discount)
Spot L4: $0.266/hour (70% discount)
```

### Cost Tracking with Labels

**Apply labels to GPU instances**:
```bash
# Create instance with cost tracking labels
gcloud compute instances create gpu-training-01 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --labels=project=nlp-training,team=research,environment=production,cost-center=ml-ops

# Query costs by label (Cloud Billing export to BigQuery required)
bq query --use_legacy_sql=false '
SELECT
  labels.value AS team,
  SUM(cost) AS total_cost_usd,
  SUM(usage.amount) AS gpu_hours
FROM `project.dataset.gcp_billing_export`
WHERE sku.description LIKE "%GPU%"
  AND labels.key = "team"
GROUP BY team
ORDER BY total_cost_usd DESC
'
```

**Grafana cost dashboard**:
```promql
# Estimated cost per GPU (requires cost metric exported to Prometheus)
# Assuming A100 at $3.673/hour
sum(DCGM_FI_DEV_GPU_UTIL > 0) * 3.673 / 60  # Cost per minute

# Monthly cost projection
sum_over_time((DCGM_FI_DEV_GPU_UTIL > 0)[30d]) * 3.673 / 60 * 60 * 24 * 30
```

**Cloud Monitoring cost alert**:
```bash
# Alert when monthly GPU spend exceeds budget
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="GPU Budget Exceeded" \
  --condition-display-name="Monthly GPU cost > $10,000" \
  --condition-threshold-value=10000 \
  --condition-threshold-duration=86400s \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-resource-type=billing_account \
  --condition-threshold-metric-filter='metric.type="billing.googleapis.com/cost" AND resource.labels.sku_description=~".*GPU.*"'
```

---

## Section 8: arr-coc-0-1 Monitoring Setup

**Production monitoring stack** for the arr-coc-0-1 training infrastructure.

### Architecture

```
Compute Engine (8×A100)
├── DCGM Daemon (collects GPU metrics)
├── Ops Agent (sends metrics to Cloud Monitoring)
└── DCGM Exporter (exports to Prometheus)
    ↓
Cloud Monitoring (dashboards + alerts)
    ↓
Prometheus (time-series storage, 30-day retention)
    ↓
Grafana (visualization + cost tracking)
    ↓
Alertmanager (PagerDuty, Slack, Email)
```

### Implementation

**1. Deploy monitoring stack** (Terraform):
```hcl
# monitoring.tf
resource "google_compute_instance" "arr_coc_gpu_training" {
  name         = "arr-coc-training-01"
  machine_type = "a2-highgpu-8g"  # 8×A100 80GB
  zone         = "us-central1-a"

  guest_accelerator {
    type  = "nvidia-tesla-a100"
    count = 8
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    # Install NVIDIA drivers + DCGM + Ops Agent
    /opt/deeplearning/install-driver.sh
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt update && apt install -y datacenter-gpu-manager
    systemctl --now enable nvidia-dcgm

    # Install Ops Agent
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
    bash add-google-cloud-ops-agent-repo.sh --also-install

    # Configure Ops Agent for DCGM
    cat > /etc/google-cloud-ops-agent/config.yaml <<YAML
    metrics:
      receivers:
        dcgm:
          type: dcgm
          collection_interval: 30s
      service:
        pipelines:
          dcgm:
            receivers: [dcgm]
    YAML
    service google-cloud-ops-agent restart
  EOF

  labels = {
    project      = "arr-coc-vision"
    environment  = "training"
    cost-center  = "ml-research"
    gpu-type     = "a100-80gb"
  }
}

# Cloud Monitoring alert policy
resource "google_monitoring_alert_policy" "gpu_idle" {
  display_name = "arr-coc GPU Idle Alert"
  combiner     = "OR"
  conditions {
    display_name = "GPU utilization < 10% for 1 hour"
    condition_threshold {
      filter          = "metric.type=\"compute.googleapis.com/instance/gpu/utilization\" resource.type=\"gce_instance\""
      duration        = "3600s"
      comparison      = "COMPARISON_LT"
      threshold_value = 0.10
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.name]
}
```

**2. Grafana dashboard JSON** (arr-coc-gpu-dashboard.json):
```json
{
  "dashboard": {
    "title": "arr-coc-0-1 GPU Training Monitoring",
    "panels": [
      {
        "title": "GPU Utilization (8×A100)",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL{instance=\"arr-coc-training-01\"}",
            "legendFormat": "GPU {{gpu}}"
          }
        ],
        "type": "graph",
        "yaxes": [{"format": "percent", "max": 100}]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_FB_USED{instance=\"arr-coc-training-01\"} / 1024",
            "legendFormat": "GPU {{gpu}} Memory (GB)"
          }
        ],
        "type": "graph",
        "yaxes": [{"format": "decgbytes", "max": 80}]
      },
      {
        "title": "GPU Temperature",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_TEMP{instance=\"arr-coc-training-01\"}",
            "legendFormat": "GPU {{gpu}} Temp"
          }
        ],
        "type": "graph",
        "yaxes": [{"format": "celsius", "max": 90}],
        "alert": {
          "name": "GPU Overheating",
          "conditions": [{"evaluator": {"params": [85], "type": "gt"}}]
        }
      },
      {
        "title": "Training Cost (Estimated)",
        "targets": [
          {
            "expr": "sum(DCGM_FI_DEV_GPU_UTIL > 0) * 3.673 / 60",
            "legendFormat": "Cost per minute ($)"
          }
        ],
        "type": "singlestat",
        "format": "currencyUSD"
      }
    ]
  }
}
```

**3. Training job instrumentation** (Python):
```python
# training/monitor.py
import torch
import time
from prometheus_client import start_http_server, Gauge, Counter

# Prometheus metrics
batch_duration = Gauge('training_batch_duration_seconds', 'Time per batch')
gpu_memory_used = Gauge('training_gpu_memory_gb', 'GPU memory used', ['gpu_id'])
batches_completed = Counter('training_batches_total', 'Total batches completed')
epoch_loss = Gauge('training_epoch_loss', 'Loss per epoch')

# Start Prometheus exporter
start_http_server(8000)

# Training loop with monitoring
for epoch in range(num_epochs):
    epoch_start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()

        # Training step
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Record metrics
        batch_duration.set(time.time() - batch_start)
        batches_completed.inc()

        # GPU memory monitoring
        for i in range(torch.cuda.device_count()):
            memory_gb = torch.cuda.memory_allocated(i) / 1e9
            gpu_memory_used.labels(gpu_id=i).set(memory_gb)

    # Epoch metrics
    epoch_loss.set(loss.item())
    print(f"Epoch {epoch}: Loss={loss.item():.4f}, Duration={time.time()-epoch_start:.2f}s")
```

**4. Alert rules** (alert-rules-arr-coc.yml):
```yaml
groups:
- name: arr_coc_training_alerts
  interval: 30s
  rules:
  - alert: ArrCocGPUWaste
    expr: avg_over_time(DCGM_FI_DEV_GPU_UTIL{instance="arr-coc-training-01"}[1h]) < 10
    for: 1h
    labels:
      severity: warning
      cost_impact: high
    annotations:
      summary: "arr-coc-0-1 GPU idle - wasting $29.38/hour (8×A100)"
      description: "GPU utilization: {{ $value }}%. Consider pausing training or terminating instance."

  - alert: ArrCocTrainingStalled
    expr: rate(training_batches_total[10m]) == 0
    for: 15m
    labels:
      severity: critical
    annotations:
      summary: "arr-coc-0-1 training stalled for 15 minutes"
      description: "No batch completions detected. GPU may be hung or process crashed."

  - alert: ArrCocMemoryLeak
    expr: deriv(training_gpu_memory_gb[1h]) > 0.5
    for: 2h
    labels:
      severity: warning
    annotations:
      summary: "arr-coc-0-1 potential memory leak detected"
      description: "GPU memory increasing by {{ $value | humanize }}GB/hour. Check for tensor accumulation."
```

**5. Cost tracking query** (BigQuery):
```sql
-- Monthly GPU spend for arr-coc-0-1 project
SELECT
  DATE_TRUNC(usage_start_time, MONTH) AS month,
  SUM(cost) AS total_cost_usd,
  SUM(usage.amount) AS total_gpu_hours,
  ROUND(SUM(cost) / SUM(usage.amount), 2) AS avg_cost_per_hour
FROM `project.dataset.gcp_billing_export`
WHERE sku.description LIKE '%NVIDIA Tesla A100%'
  AND labels.value = 'arr-coc-vision'  -- project label
GROUP BY month
ORDER BY month DESC

-- Output example:
-- month        | total_cost_usd | total_gpu_hours | avg_cost_per_hour
-- 2025-11-01   | $17,534.40     | 4,776           | $3.67
-- 2025-10-01   | $22,018.56     | 6,000           | $3.67
```

---

## Sources

**Google Cloud Documentation:**
- [Cloud Monitoring Ops Agent DCGM Integration](https://docs.cloud.google.com/monitoring/agent/ops-agent/third-party-nvidia) (accessed 2025-11-16)
- [Google Cloud Observability Release Notes](https://docs.cloud.google.com/stackdriver/docs/release-notes) (accessed 2025-11-16)
- [Committed Use Discounts Overview](https://docs.cloud.google.com/compute/docs/instances/committed-use-discounts-overview) (accessed 2025-11-16)
- [GPU Pricing](https://cloud.google.com/compute/gpus-pricing) (accessed 2025-11-16)

**NVIDIA Documentation:**
- [NVIDIA Data Center GPU Manager (DCGM)](https://developer.nvidia.com/dcgm) (accessed 2025-11-16)
- [NVIDIA GPU Telemetry](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/index.html) (November 25, 2024)
- [NVIDIA Compute Sanitizer Blog Post](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/) (June 29, 2023)

**Grafana:**
- [Grafana GPU Observability Setup](https://grafana.com/docs/grafana-cloud/monitor-applications/ai-observability/gpu-observability/setup/) (accessed 2025-11-16)
- [Grafana GPU Monitoring Dashboard 9822](https://grafana.com/grafana/dashboards/9822-gpu-monitoring/) (accessed 2025-11-16)

**Community & Tutorials:**
- [Medium: Self-managed GPU Monitoring Stack on Google Cloud with DCGM, Prometheus, and Grafana](https://medium.com/google-cloud/self-managed-gpu-monitoring-stack-on-google-cloud-with-dcgm-prometheus-and-grafana-04c8355c5132) (Xiang Shen, September 16, 2023)
- [LeaderGPU: Collecting GPU Metrics with Grafana](https://www.leadergpu.com/articles/524-collecting-gpu-metrics-with-grafana) (June 25, 2024)
- [PyTorch Forums: How to debug causes of GPU memory leaks?](https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741) (August 26, 2017)
- [GitHub: NVIDIA/dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) (accessed 2025-11-16)
- [GitHub: utkuozdemir/nvidia_gpu_exporter](https://github.com/utkuozdemir/nvidia_gpu_exporter) (accessed 2025-11-16)

**Additional References:**
- [CloudBolt: GCP Discounts Guide](https://www.cloudbolt.io/gcp-cost-optimization/gcp-discounts/) (accessed 2025-11-16)
- [Economize Cloud: GCP Cloud GPUs – Pricing & Discounts Comparison](https://www.economize.cloud/blog/gcp-gpu-pricing-comparison/) (September 27, 2024)
- [Massed Compute: GPU Monitoring FAQs](https://massedcompute.com/faq-answers/?question=How%20to%20monitor%20GPU%20utilization%20in%20real-time%20using%20Prometheus%20and%20Grafana%20in%20a%20Kubernetes%20cluster?) (accessed 2025-11-16)
