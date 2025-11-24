# Production Deployment Troubleshooting: Containers, Drivers, Multi-Tenant GPU

## Overview

Production GPU deployments introduce unique challenges beyond development environments: container runtime integration, driver compatibility across host/container boundaries, multi-tenant resource isolation, and orchestration platform complexities. This document covers expert-level troubleshooting for NVIDIA Container Toolkit, driver version mismatches, MPS/MIG configuration issues, and Kubernetes GPU scheduling problems.

**Why Production Deployment is Hard:**
- Container runtimes abstract GPU access through multiple layers (docker → nvidia-container-runtime → libnvidia-container → driver)
- Driver versions must be compatible across host, container toolkit, and CUDA toolkit
- Multi-tenant scenarios require careful isolation without performance degradation
- Orchestrators like Kubernetes add scheduling and resource management complexity
- SELinux, cgroups, and device permissions create security vs functionality tradeoffs

From [NVIDIA Container Toolkit Troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html) (accessed 2025-11-13):
> "When using the NVIDIA Container Runtime Hook to inject requested GPUs and driver libraries into a container, the hook makes modifications without the low-level runtime (such as runc) being aware of these changes."

**Related Knowledge:**
- See [cuda/09-runtime-errors-debugging-expert.md](09-runtime-errors-debugging-expert.md) for CUDA runtime error debugging
- See [cuda/02-pytorch-build-system-compilation.md](02-pytorch-build-system-compilation.md) for build-time driver considerations
- See [cuda/11-advanced-troubleshooting-multi-gpu-expert.md](11-advanced-troubleshooting-multi-gpu-expert.md) for multi-GPU debugging

---

## Section 1: NVIDIA Container Toolkit & Docker Runtime Issues (~125 lines)

### Understanding the Container Stack

The GPU container stack has multiple components that must work together:

```
Application (PyTorch/TensorFlow)
        ↓
    CUDA Runtime
        ↓
libnvidia-container (injects drivers)
        ↓
nvidia-container-runtime-hook
        ↓
Docker/containerd + runc
        ↓
    Host NVIDIA Driver
        ↓
    GPU Hardware
```

From [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html) (accessed 2025-11-13):
> "For most common issues, you can generate debugging logs to help identify the root cause of the problem. To generate debug logs: Edit your runtime configuration under `/etc/nvidia-container-runtime/config.toml` and uncomment the `debug=...` line."

### Common Container Toolkit Errors

**Error 1: "Failed to initialize NVML: Unknown Error"**

This occurs when containers lose GPU access after updates or systemd reloads.

```bash
# Symptoms
docker run --gpus all nvidia/cuda:11.8.0-base nvidia-smi
# Failed to initialize NVML: Unknown Error

# Root cause: Container update removed cgroup access
# Solution 1: Use cgroupfs driver instead of systemd
cat /etc/docker/daemon.json
{
  "exec-opts": ["native.cgroupdriver=cgroupfs"]
}
sudo systemctl restart docker

# Solution 2: Explicitly request device nodes
docker run --gpus all \
  --device=/dev/nvidia0 \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm \
  nvidia/cuda:11.8.0-base nvidia-smi

# Solution 3: Use CDI (Container Device Interface) - recommended
# CDI includes device nodes in container config automatically
```

**Error 2: "nvidia-container-cli: initialization error"**

```bash
# Symptoms
docker: Error response from daemon: failed to create shim task:
OCI runtime create failed: runc create failed:
unable to start container process: error during container init:
error running hook #0: error running hook: exit status 1

# Debug with verbose logging
sudo docker run --rm --gpus all \
  --runtime=nvidia \
  nvidia/cuda:11.8.0-base nvidia-smi 2>&1 | tee container-debug.log

# Check nvidia-container-runtime-hook logs
sudo journalctl -u docker.service | grep nvidia-container

# Common fix: Reinstall container toolkit
sudo apt-get purge -y nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Error 3: GPU Not Visible in Container**

From [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/cuda-11-4-2-docker-image-driver-version-mismatch/198833) (accessed 2025-11-13):

```bash
# Inside container, check driver version
nvidia-smi
# CUDA Version: 11.4 Driver Version: 470.57.02

# On host, check actual driver
nvidia-smi
# CUDA Version: 12.2 Driver Version: 535.86.10

# Diagnosis: Container sees old driver through libnvidia-container
# Check container toolkit version
dpkg -l | grep nvidia-container-toolkit

# Update to latest toolkit for driver compatibility
sudo apt-get update
sudo apt-get install --only-upgrade nvidia-container-toolkit
```

### Driver/Container Version Compatibility

**Forward Compatibility Matrix:**

| Host Driver | Compatible CUDA Toolkit Versions |
|-------------|----------------------------------|
| 535.x | CUDA 12.2, 12.1, 12.0, 11.x |
| 525.x | CUDA 12.0, 11.x |
| 520.x | CUDA 11.8, 11.7, 11.x |
| 470.x | CUDA 11.4, 11.3, 11.2, 11.x |

From [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) (accessed 2025-11-13):
> "CUDA forward compatibility allows CUDA applications built against older CUDA toolkits to run on systems with newer NVIDIA drivers."

**Checking Compatibility:**

```bash
# On host: Check driver version
nvidia-smi | grep "Driver Version"

# In container: Check CUDA version
cat /usr/local/cuda/version.txt

# Verify compatibility
# Rule: Host driver version >= Required driver for CUDA version
# CUDA 11.8 requires driver >= 450.80.02
# CUDA 12.0 requires driver >= 525.60.13
# CUDA 12.2 requires driver >= 535.54.03
```

### SELinux and Permission Issues

From [NVIDIA Container Toolkit Troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html) (accessed 2025-11-13):
> "Permission denied error when running the nvidia-docker wrapper under SELinux. When running on SELinux environments one may see: `/bin/nvidia-docker: line 34: /bin/docker: Permission denied`"

```bash
# Solution 1: Specify runtime directly
sudo docker run --gpus=all --runtime=nvidia --rm \
  nvidia/cuda:11.8.0-base nvidia-smi

# Solution 2: Generate SELinux policy
ausearch -c 'nvidia-docker' --raw | audit2allow -M my-nvidiadocker
semodule -X 300 -i my-nvidiadocker.pp

# Solution 3: Disable SELinux separation (reduces security)
docker run --security-opt=label=disable \
  --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

**NVML Insufficient Permissions on RHEL/CentOS:**

```bash
# Error: Failed to initialize NVML: Insufficient Permissions
# Solution: Disable SELinux separation or adjust policy
docker run --security-opt=label=disable \
  --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

---

## Section 2: GPU Device Access & System-Level Issues (~125 lines)

### Device Node Permissions

GPU access requires correct device node permissions:

```bash
# Check GPU device nodes
ls -la /dev/nvidia*
crw-rw-rw- 1 root root 195,   0 Nov 13 10:00 /dev/nvidia0
crw-rw-rw- 1 root root 195,   1 Nov 13 10:00 /dev/nvidia1
crw-rw-rw- 1 root root 195, 255 Nov 13 10:00 /dev/nvidiactl
crw-rw-rw- 1 root root 195, 254 Nov 13 10:00 /dev/nvidia-uvm
crw-rw-rw- 1 root root 237,   0 Nov 13 10:00 /dev/nvidia-uvm-tools

# Fix permissions if incorrect
sudo chmod 666 /dev/nvidia*
sudo chmod 666 /dev/nvidia-uvm*

# Make permanent with udev rules
cat /etc/udev/rules.d/70-nvidia.rules
KERNEL=="nvidia", RUN+="/bin/bash -c '/usr/bin/nvidia-smi -L && /bin/chmod 666 /dev/nvidia*'"
KERNEL=="nvidia_uvm", RUN+="/bin/bash -c '/usr/bin/nvidia-modprobe -c0 -u && /bin/chmod 0666 /dev/nvidia-uvm*'"

sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Cgroup Device Access

Containers use cgroups to control device access:

```bash
# Check if GPU devices are accessible to cgroup
cat /sys/fs/cgroup/devices/docker/<container-id>/devices.list | grep nvidia
c 195:* rwm  # GPU devices
c 237:* rwm  # nvidia-uvm

# Debug cgroup issues
docker inspect <container-id> | grep -A 20 Devices

# Verify device mounting
docker run --gpus all nvidia/cuda:11.8.0-base ls -la /dev/nvidia*
```

### Missing /dev/char Symlinks

From [NVIDIA Container Toolkit GitHub Discussion #1133](https://github.com/NVIDIA/nvidia-container-toolkit/discussions/1133) (accessed 2025-11-13):
> "Certain runc versions show similar behavior with the systemd cgroup driver when /dev/char symlinks for the required devices are missing on the system."

```bash
# Check for /dev/char symlinks
ls -la /dev/char/
lrwxrwxrwx 1 root root 11 Nov 13 10:00 195:0 -> ../nvidia0
lrwxrwxrwx 1 root root 11 Nov 13 10:00 195:1 -> ../nvidia1
lrwxrwxrwx 1 root root 14 Nov 13 10:00 195:255 -> ../nvidiactl
lrwxrwxrwx 1 root root 16 Nov 13 10:00 237:0 -> ../nvidia-uvm

# Create missing symlinks
sudo mkdir -p /dev/char
for dev in /dev/nvidia*; do
  major=$(stat -c '%t' $dev)
  minor=$(stat -c '%T' $dev)
  sudo ln -sf ../${dev##*/} /dev/char/$((0x$major)):$((0x$minor))
done
```

### Driver Module Loading Issues

```bash
# Check if NVIDIA driver is loaded
lsmod | grep nvidia
nvidia_uvm            1310720  0
nvidia_drm             69632  2
nvidia_modeset       1318912  2 nvidia_drm
nvidia              56549376  91 nvidia_uvm,nvidia_modeset

# Driver not loaded - load manually
sudo modprobe nvidia
sudo modprobe nvidia-uvm

# Check for module loading errors
dmesg | grep -i nvidia

# Common issue: Nouveau conflict
# Blacklist nouveau driver
cat /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0

sudo update-initramfs -u
sudo reboot
```

### Host-Container Driver Mismatch Debugging

```bash
# Host driver version
cat /sys/module/nvidia/version
535.86.10

# Container sees driver through bind-mounts
docker run --gpus all nvidia/cuda:11.8.0-base cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.86.10

# Check libnvidia-container binding
nvidia-container-cli info
NVRM version:   535.86.10
CUDA version:   12.2

Driver version: 535.86.10
Device #0:      NVIDIA A100-SXM4-40GB

# Debug missing library bindings
docker run --gpus all nvidia/cuda:11.8.0-base ldconfig -p | grep cuda
libcudart.so.11.0 -> /usr/local/cuda-11.8/lib64/libcudart.so.11.0
```

---

## Section 3: Multi-Tenant GPU Isolation (MPS & MIG) (~125 lines)

### MPS (Multi-Process Service) Configuration

From [NVIDIA Developer Forums - CUDA MPS Multi-GPU](https://forums.developer.nvidia.com/t/cuda-mps-not-working-as-expected-in-multi-gpu-environment/312599) (accessed 2025-11-13):
> "MPS can be configured to use multiple GPUs, but when multiple GPUs are visible to the MPS server/daemon, there is no automatic distribution system to route different jobs to different GPUs."

**MPS Fundamentals:**

```bash
# Start MPS server (per GPU)
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Verify MPS is running
ps aux | grep mps-server
# nvidia-cuda-mps-server

# Check MPS active clients
echo get_server_list | nvidia-cuda-mps-control
server_pid 12345 /dev/nvidia0 active
```

**MPS Multi-GPU Limitation:**

From [GPU Sharing That Works: MIG, MPS & Schedulers](https://medium.com/@hadiyolworld007/gpu-sharing-that-works-mig-mps-schedulers-b3105933d1aa) (accessed 2025-11-13):
> "MPS (Multi-Process Service): soft sharing of one GPU among many CUDA contexts by merging them into one server process to lower context switching and enable overlapping kernels. Great for many small requests."

```bash
# Problem: MPS doesn't automatically distribute across GPUs
# All jobs go to first visible GPU

# Solution: Launch separate MPS daemon per GPU
for gpu_id in 0 1 2 3; do
  export CUDA_VISIBLE_DEVICES=$gpu_id
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$gpu_id
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$gpu_id
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d
done

# Verify multiple MPS servers
nvidia-smi | grep mps-server
#   0  N/A  N/A  12345    C   nvidia-cuda-mps-server    30MiB
#   1  N/A  N/A  12346    C   nvidia-cuda-mps-server    30MiB
#   2  N/A  N/A  12347    C   nvidia-cuda-mps-server    30MiB
#   3  N/A  N/A  12348    C   nvidia-cuda-mps-server    30MiB
```

**MPS Thread Percentage Configuration:**

```bash
# Limit each client to 25% of GPU resources
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

# Verify client allocation
echo get_default_active_thread_percentage | nvidia-cuda-mps-control
25

# Set per-client limits
echo set_active_thread_percentage <pid> 50 | nvidia-cuda-mps-control
```

**MPS Error Isolation Issues:**

```bash
# Problem: One client crashes, entire MPS server goes down
# Solution: Use Volta+ MPS with client isolation

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
compute_cap
7.0   # Volta (sm_70) - has MPS isolation
8.0   # Ampere (sm_80) - has MPS isolation
6.1   # Pascal (sm_61) - NO isolation

# Volta+ MPS features:
# - Per-client GPU memory protection
# - Address space isolation
# - Fault containment (crash doesn't affect other clients)
```

### MIG (Multi-Instance GPU) Configuration

From [NVIDIA Multi-Instance GPU User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) (accessed 2025-11-13):
> "The Multi-Instance GPU (MIG) User Guide explains how to partition supported NVIDIA GPUs into multiple isolated instances, each with dedicated compute and memory resources."

**MIG Supported GPUs:**
- A100 (all variants)
- A30
- H100
- H200 (announced)

**Enable MIG Mode:**

```bash
# Check if MIG is supported
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv
mig.mode.current
Disabled

# Enable MIG mode (requires reboot/GPU reset)
sudo nvidia-smi -i 0 -mig 1
# Reboot required. To change the MIG mode, please reboot the system.

sudo reboot

# Verify MIG enabled
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv
mig.mode.current
Enabled
```

**Create MIG Instances:**

```bash
# List available MIG profiles
nvidia-smi mig -lgip
# Profile ID, Profile Name, Memory Size
# 0           1g.5gb        5GB
# 9           1g.10gb      10GB
# 5           2g.10gb      10GB
# 14          3g.20gb      20GB
# 19          4g.20gb      20GB
# 15          7g.40gb      40GB

# Create MIG instance (3g.20gb = 3 GPU slices, 20GB memory)
sudo nvidia-smi mig -cgi 14 -C
# Successfully created GPU instance ID  1 on GPU  0 using profile MIG 3g.20gb
# Successfully created compute instance ID  0 on GPU instance ID  1

# List created instances
nvidia-smi mig -lgi
# GPU instance ID, Name, Profile ID, Placement
# 1              3g.20gb      14       {0,1,2}

# Destroy MIG instance
sudo nvidia-smi mig -dci -ci 0 -gi 1  # Destroy compute instance
sudo nvidia-smi mig -dgi -gi 1        # Destroy GPU instance
```

**MIG vs MPS Decision Tree:**

From [GPU Sharing That Works: MIG, MPS & Schedulers](https://medium.com/@hadiyolworld007/gpu-sharing-that-works-mig-mps-schedulers-b3105933d1aa) (accessed 2025-11-13):
> "MIG (Multi-Instance GPU): hard partitioning of a data-center GPU into isolated slices — each with dedicated SMs, memory bandwidth, and VRAM. Great for noisy neighbors and predictable latency."

```
Use MIG when:
- Need strict isolation (noisy neighbor protection)
- Predictable latency/bandwidth SLAs required
- Different tenants/security domains
- Large workloads (each slice gets significant resources)

Use MPS when:
- Many small concurrent jobs (100+ clients)
- Single tenant/trust domain
- Maximize utilization of idle GPU cycles
- Workloads with low kernel occupancy
```

---

## Section 4: Kubernetes GPU Scheduling & Production Operations (~125 lines)

### Kubernetes GPU Device Plugin

From [Kubernetes Schedule GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) (accessed 2025-11-13):
> "Kubernetes includes stable support for managing AMD and NVIDIA GPUs across different nodes in your cluster, using device plugins."

**Install NVIDIA Device Plugin:**

```bash
# Using Helm
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --set-string devicePlugin.version=v0.14.0

# Verify plugin running
kubectl get pods -n kube-system | grep nvidia-device-plugin
nvidia-device-plugin-xxxx   1/1     Running   0          1m

# Check GPU resources advertised
kubectl get nodes -o json | jq '.items[].status.capacity'
{
  "nvidia.com/gpu": "8",
  "cpu": "96",
  "memory": "900Gi"
}
```

**Request GPUs in Pod:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:11.8.0-base
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
    command: ["nvidia-smi"]
  restartPolicy: Never
```

**GPU Scheduling Errors:**

```bash
# Error: "0/3 nodes are available: insufficient nvidia.com/gpu"
kubectl describe pod gpu-pod
# Events:
#   Warning  FailedScheduling  pod/gpu-pod (0/3 nodes available:
#   3 Insufficient nvidia.com/gpu)

# Debug: Check GPU availability
kubectl get nodes -o json | \
  jq '.items[] | {name: .metadata.name, gpus: .status.capacity["nvidia.com/gpu"]}'

# Check if GPUs are already allocated
kubectl get pods -o json --all-namespaces | \
  jq '.items[] | select(.spec.containers[].resources.limits["nvidia.com/gpu"] != null) |
  {name: .metadata.name, namespace: .metadata.namespace, gpus: .spec.containers[].resources.limits["nvidia.com/gpu"]}'
```

### GPU Time-Slicing in Kubernetes

From [Oversubscribing GPUs in Kubernetes](https://jacobtomlinson.dev/posts/2023/oversubscribing-gpus-in-kubernetes/) (accessed 2025-11-13):
> "In this post we are going to use time slicing to share our GPUs between Pods. This works by running many CUDA processes on the same GPU and giving them equal time slices."

```bash
# Create ConfigMap for time-slicing
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: kube-system
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        replicas: 4  # Allow 4 pods per GPU
EOF

# Update device plugin with time-slicing
helm upgrade nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --set-string config.name=time-slicing-config

# Verify: Now each GPU advertises 4 slots
kubectl get nodes -o json | jq '.items[].status.capacity'
{
  "nvidia.com/gpu": "32",  # 8 GPUs × 4 replicas
  "cpu": "96",
  "memory": "900Gi"
}
```

### Node Labeling for GPU Types

```bash
# Automatically label nodes with GPU info (using Node Feature Discovery)
kubectl apply -k https://github.com/kubernetes-sigs/node-feature-discovery/deployment/overlays/default

# Verify GPU labels
kubectl get nodes -o json | \
  jq '.items[] | {name: .metadata.name, labels: .metadata.labels |
  with_entries(select(.key | startswith("feature.node.kubernetes.io/pci-10de")))}'

# Schedule pod to specific GPU type
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod-a100
spec:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values: ["NVIDIA-A100-SXM4-40GB"]
  containers:
  - name: cuda-container
    image: nvidia/cuda:11.8.0-base
    resources:
      limits:
        nvidia.com/gpu: 1
```

### Production Monitoring and Health Checks

```bash
# GPU node health check script
cat > /usr/local/bin/gpu-health-check.sh <<'EOF'
#!/bin/bash
# Exit non-zero if GPU not accessible
nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE | grep -q "GPU 0"
exit $?
EOF
chmod +x /usr/local/bin/gpu-health-check.sh

# Kubernetes liveness probe
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  containers:
  - name: app
    image: my-gpu-app:latest
    resources:
      limits:
        nvidia.com/gpu: 1
    livenessProbe:
      exec:
        command:
        - /bin/sh
        - -c
        - "nvidia-smi -q -d MEMORY,UTILIZATION | grep -q 'GPU 0'"
      initialDelaySeconds: 30
      periodSeconds: 60
```

**Production Runbook: GPU Not Available in Pod**

```bash
# Step 1: Verify GPU on node
kubectl debug node/<node-name> -it --image=ubuntu
# (inside debug pod)
chroot /host
nvidia-smi

# Step 2: Check device plugin logs
kubectl logs -n kube-system -l app=nvidia-device-plugin --tail=100

# Step 3: Verify device nodes mounted in pod
kubectl exec -it <pod-name> -- ls -la /dev/nvidia*

# Step 4: Check cgroup device access
kubectl exec -it <pod-name> -- cat /sys/fs/cgroup/devices/devices.list | grep nvidia

# Step 5: Restart device plugin if needed
kubectl delete pod -n kube-system -l app=nvidia-device-plugin
```

**Emergency Recovery: GPU Reset**

```bash
# Drain node (moves workloads off)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# SSH to node and reset GPU
ssh <node-name>
sudo nvidia-smi --gpu-reset

# Reload driver if needed
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia-uvm

# Restart NVIDIA persistence daemon
sudo systemctl restart nvidia-persistenced

# Uncordon node
kubectl uncordon <node-name>

# Verify pods scheduled back
kubectl get pods -o wide | grep <node-name>
```

---

## Sources

**Container Toolkit Documentation:**
- [NVIDIA Container Toolkit Troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html) - Official troubleshooting guide (accessed 2025-11-13)

**Multi-Tenant GPU Isolation:**
- [NVIDIA Multi-Instance GPU User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) - MIG configuration and deployment (accessed 2025-11-13)
- [GPU Sharing That Works: MIG, MPS & Schedulers](https://medium.com/@hadiyolworld007/gpu-sharing-that-works-mig-mps-schedulers-b3105933d1aa) - MIG vs MPS comparison (accessed 2025-11-13)
- [NVIDIA Developer Forums - CUDA MPS Multi-GPU](https://forums.developer.nvidia.com/t/cuda-mps-not-working-as-expected-in-multi-gpu-environment/312599) - MPS multi-GPU limitations (accessed 2025-11-13)

**Kubernetes GPU Scheduling:**
- [Kubernetes Schedule GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) - Official Kubernetes GPU documentation (accessed 2025-11-13)
- [Oversubscribing GPUs in Kubernetes](https://jacobtomlinson.dev/posts/2023/oversubscribing-gpus-in-kubernetes/) - Time-slicing implementation (accessed 2025-11-13)

**Community Discussions:**
- [NVIDIA Container Toolkit GitHub Discussion #1133](https://github.com/NVIDIA/nvidia-container-toolkit/discussions/1133) - /dev/char symlinks issue (accessed 2025-11-13)
- [NVIDIA Developer Forums - Driver Version Mismatch](https://forums.developer.nvidia.com/t/cuda-11-4-2-docker-image-driver-version-mismatch/198833) - Container driver compatibility (accessed 2025-11-13)

**Additional References:**
- NVIDIA CUDA Compatibility Documentation
- Docker cgroup device access
- Kubernetes Node Feature Discovery
