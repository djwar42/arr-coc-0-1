# XOTorch - ML Inference Engine TUI

## Overview

XOTorch is a distributed inference and training framework focused on PyTorch, featuring a terminal-based chat interface built with Rich (not Textual). This case study demonstrates real-time ML inference visualization, distributed computing coordination, and token-by-token streaming output in a terminal UI.

**Project**: [shamantechnology/xotorch](https://github.com/shamantechnology/xotorch)
**Fork of**: [Exo v1](https://github.com/exo-explore/exo) (hard fork continuing open source development)
**Tech Stack**: PyTorch, Rich library, asyncio, GRPC, distributed computing
**License**: GPL-3.0
**Stars**: 28 | **Forks**: 5

## Project Background

From [XOTorch README](https://github.com/shamantechnology/xotorch/blob/main/README.md) (accessed 2025-11-02):

> The focus on xotorch, and other xo* projects, is more about minimization and focusing on individual Tensor or machine learning libraries instead of all in one. This project is focused on using pytorch and torchtune as, with building out the pytorch inference engine, pytorch has some better reach on running on different platforms.

**Key Decision**: Hard fork from Exo v1 after upstream moved to closed-source v2, to continue open source distributed inference development.

## Architecture Overview

### Distributed Inference System

From [main.py](https://github.com/shamantechnology/xotorch/blob/main/xotorch/main.py) (accessed 2025-11-02):

**Core Components**:
- **Node**: Distributed computation node with GRPC communication
- **Topology**: Network visualization of connected inference nodes
- **Shard Downloader**: Handles model weight distribution across nodes
- **Inference Engine**: PyTorch-based model execution
- **TUI Visualization**: Real-time network and chat interface

**Event Loop Configuration**:
```python
# Lines 48-67: High-performance async setup
def configure_uvloop():
    if use_win:
      winloop.install()
    else:
      uvloop.install()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Increase file descriptor limits on Unix systems
    if not psutil.WINDOWS:
      soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
      try: resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
      except ValueError:
        try: resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
        except ValueError: pass

    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) * 4)
        )
    )
    return loop
```

**Pattern**: Cross-platform event loop optimization with uvloop (Unix) and winloop (Windows) for maximum async performance.

## Terminal UI Implementation

### 1. Chat TUI (Simple Text Interface)

From [chat_tui.py](https://github.com/shamantechnology/xotorch/blob/main/xotorch/viz/chat_tui.py) (accessed 2025-11-02):

**Simple ASCII interface for ML inference**:
```python
# Lines 28-36: Terminal interface header
print("\n╔══════════════════════════════════════════════════════════════╗")
print("║             XOTORCH TERMINAL INTERFACE                       ║")
print("╠══════════════════════════════════════════════════════════════╣")
print("║ Type your prompt after the '>' prompt below                  ║")
print("║ Commands: 'exit' to quit, 'model <name>' to switch models    ║")
print("║ Supported models: llama-3.2-1b, llama-3.2-3b, llama-3-8b     ║")
print("╚══════════════════════════════════════════════════════════════╝")
```

**Token Streaming with Performance Metrics**:
```python
# Lines 78-124: Real-time token tracking
async def track_token_speed():
    nonlocal tokens_per_second, last_token_count
    tokens = []
    full_response = ""

    def on_token(_request_id, _tokens, _is_finished):
        nonlocal tokens_per_second, last_token_count, start_time, full_response
        tokens.extend(_tokens)

        # Decode and display tokens as they arrive
        try:
            if _tokens:
                tokenizer = node.inference_engine.tokenizer
                new_text = tokenizer.decode(_tokens)
                full_response += new_text
        except Exception as e:
            if DEBUG >= 2:
                print(f"\nError decoding tokens: {e}")

        return _request_id == request_id and _is_finished

    try:
        await callback.wait(on_token, timeout=300)

        # Calculate final metrics
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed > 0:
            tokens_per_second = len(tokens) / elapsed

        # Display completion stats
        tflops = node.topology.nodes.get(
            node.id, UNKNOWN_DEVICE_CAPABILITIES
        ).flops.fp16
        print(f"Final stats: {len(tokens)} tokens | "
              f"{tokens_per_second:.2f} tokens/sec | {tflops:.2f} TFLOPS\n")
```

**Pattern**: Event-based token streaming with async callbacks for real-time performance monitoring.

### 2. Topology Visualization (Rich Library)

From [topology_viz.py](https://github.com/shamantechnology/xotorch/blob/main/xotorch/viz/topology_viz.py) (accessed 2025-11-02):

**Rich Live Display for Network Topology**:
```python
# Lines 20-41: Multi-panel Rich layout
class TopologyViz:
    def __init__(self, chatgpt_api_endpoints: List[str] = [],
                 web_chat_urls: List[str] = []):
        self.console = Console()
        self.layout = Layout()

        # Three-panel layout: main topology, chat, downloads
        self.layout.split(
            Layout(name="main"),
            Layout(name="prompt_output", size=15),
            Layout(name="download", size=25)
        )

        self.main_panel = Panel(
            self._generate_main_layout(),
            title="0 Node Cluster",
            border_style="red1"
        )
        self.prompt_output_panel = Panel(
            "",
            title="Prompt and Output",
            border_style="deep_pink4"
        )
        self.download_panel = Panel(
            "",
            title="Download Progress",
            border_style="bright_white"
        )

        # Initially hide prompt_output panel
        self.layout["prompt_output"].visible = False
        self.live_panel = Live(
            self.layout,
            auto_refresh=False,
            console=self.console
        )
        self.live_panel.start()
```

**Dynamic Panel Visibility**:
```python
# Lines 66-82: Smart panel management
def refresh(self):
    self.main_panel.renderable = self._generate_main_layout()
    node_count = len(self.topology.nodes)
    self.main_panel.title = f"{node_count} Node Cluster"

    # Show/hide prompt panel based on activity
    if any(r[0] or r[1] for r in self.requests.values()):
        self.prompt_output_panel = self._generate_prompt_output_layout()
        self.layout["prompt_output"].update(self.prompt_output_panel)
        self.layout["prompt_output"].visible = True
    else:
        self.layout["prompt_output"].visible = False

    # Only show downloads panel if in progress
    if any(progress.status == "in_progress"
           for progress in self.node_download_progress.values()):
        self.download_panel.renderable = self._generate_download_layout()
        self.layout["download"].visible = True
    else:
        self.layout["download"].visible = False
```

**Pattern**: Conditional panel rendering - only show relevant information to reduce visual clutter.

### 3. Network Topology Visualization

**GPU Performance Bar**:
```python
# Lines 219-234: Visual FLOPS indicator
# Calculate total FLOPS and position on gradient bar
total_flops = sum(
    self.topology.nodes.get(
        partition.node_id, UNKNOWN_DEVICE_CAPABILITIES
    ).flops.fp16
    for partition in self.partitions
)
bar_pos = (math.tanh(total_flops**(1/3)/2.5 - 2) + 1)

# Gradient bar using emojis
gradient_bar = Text()
emojis = ["🟥", "🟧", "🟨", "🟩"]
for i in range(bar_width):
    emoji_index = min(int(i/(bar_width/len(emojis))), len(emojis) - 1)
    gradient_bar.append(emojis[emoji_index])

# Labels: "GPU poor" to "GPU rich"
```

**Circular Node Layout**:
```python
# Lines 243-291: Radial visualization of distributed nodes
for i, partition in enumerate(self.partitions):
    device_capabilities = self.topology.nodes.get(
        partition.node_id, UNKNOWN_DEVICE_CAPABILITIES
    )

    # Calculate node position on circle
    angle = 2*math.pi*i/num_partitions
    x = int(center_x + radius_x*math.cos(angle))
    y = int(center_y + radius_y*math.sin(angle))

    # Color-coded nodes
    if partition.node_id == self.topology.active_node_id:
        visualization[y][x] = "🔴"  # Active node
    elif partition.node_id == self.node_id:
        visualization[y][x] = "🟢"  # Current node
    else:
        visualization[y][x] = "🔵"  # Other nodes

    # Node info display
    node_info = [
        f"{device_capabilities.model} {device_capabilities.memory // 1024}GB",
        f"{device_capabilities.flops.fp16}TFLOPS",
        f"[{partition.start:.2f}-{partition.end:.2f}]",  # Model partition range
    ]
```

**Pattern**: Emoji-based node visualization with ASCII connections showing distributed network topology.

## ML Inference Integration

### Model Loading and Shard Management

From [main.py](https://github.com/shamantechnology/xotorch/blob/main/xotorch/main.py) (accessed 2025-11-02):

**Preemptive Model Loading**:
```python
# Lines 191-200: Background shard downloading
def preemptively_load_shard(request_id: str, opaque_status: str):
    try:
        status = json.loads(opaque_status)
        if status.get("type") != "node_status" or \
           status.get("status") != "start_process_prompt":
            return
        current_shard = node.get_current_shard(Shard.from_dict(status.get("shard")))
        if DEBUG >= 2:
            print(f"Preemptively starting download for {current_shard}")
        asyncio.create_task(node.inference_engine.ensure_shard(current_shard))
    except Exception as e:
        if DEBUG >= 2:
            print(f"Failed to preemptively start download: {e}")
            traceback.print_exc()

node.on_opaque_status.register("preemptively_load_shard").on_next(
    preemptively_load_shard
)
```

**Pattern**: Event-driven model shard loading triggered by incoming inference requests.

### Download Progress Broadcasting

**Throttled Progress Updates**:
```python
# Lines 202-212: Network-wide download status
last_events: dict[str, tuple[float, RepoProgressEvent]] = {}

def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
    global last_events
    current_time = time.time()
    if event.status == "not_started": return
    last_event = last_events.get(shard.model_id)
    if last_event and last_event[1].status == "complete" and \
       event.status == "complete": return
    if last_event and last_event[0] == event.status and \
       current_time - last_event[0] < 0.2: return
    last_events[shard.model_id] = (current_time, event)
    asyncio.create_task(
        node.broadcast_opaque_status(
            "",
            json.dumps({
                "type": "download_progress",
                "node_id": node.id,
                "progress": event.to_dict()
            })
        )
    )

shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)
```

**Pattern**: Rate-limited event broadcasting (200ms throttle) to prevent network congestion from progress updates.

### CLI Model Execution

**Direct Model Running**:
```python
# Lines 214-244: Command-line inference
async def run_model_cli(node: Node, model_name: str, prompt: str):
    inference_class = node.inference_engine.__class__.__name__
    shard = build_base_shard(model_name, inference_class)
    if not shard:
        print(f"Error: Unsupported model '{model_name}' for "
              f"inference engine {inference_class}")
        return

    tokenizer = await resolve_tokenizer(
        get_repo(shard.model_id, inference_class)
    )
    request_id = str(uuid.uuid4())
    callback_id = f"cli-wait-response-{request_id}"
    callback = node.on_token.register(callback_id)

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    try:
        print(f"Processing prompt: {prompt}")
        await node.process_prompt(shard, prompt, request_id=request_id)

        tokens = []
        def on_token(_request_id, _tokens, _is_finished):
            tokens.extend(_tokens)
            return _request_id == request_id and _is_finished

        await callback.wait(on_token, timeout=300)

        print("\nGenerated response:")
        print(tokenizer.decode(tokens))
    finally:
        node.on_token.deregister(callback_id)
```

**Pattern**: Async token collection with cleanup (deregister callback in finally block).

## Performance Optimizations

### High-Performance Async Configuration

**File Descriptor Management**:
```python
# Lines 58-63: Unix system resource limits
if not psutil.WINDOWS:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except ValueError:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
        except ValueError:
            pass
```

**Thread Pool Executor**:
```python
# Line 65: Dynamic thread pool sizing
loop.set_default_executor(
    concurrent.futures.ThreadPoolExecutor(
        max_workers=min(32, (os.cpu_count() or 1) * 4)
    )
)
```

**Pattern**: Adaptive resource allocation based on system capabilities with graceful fallback.

### Discovery and Networking

**Multiple Discovery Strategies**:
```python
# Lines 133-149: UDP vs Manual discovery
if args.discovery_module == "udp":
    discovery = UDPDiscovery(
        args.node_id,
        args.node_port,
        args.listen_port,
        args.broadcast_port,
        lambda peer_id, address, description, device_capabilities:
            GRPCPeerHandle(peer_id, address, description, device_capabilities),
        discovery_timeout=args.discovery_timeout,
        allowed_node_ids=allowed_node_ids,
        allowed_interface_types=allowed_interface_types
    )
elif args.discovery_module == "manual":
    if not args.discovery_config_path:
        raise ValueError(
            "--discovery-config-path is required when using manual discovery"
        )
    discovery = ManualDiscovery(
        args.discovery_config_path,
        args.node_id,
        create_peer_handle=lambda peer_id, address, description,
                                 device_capabilities:
            GRPCPeerHandle(peer_id, address, description, device_capabilities)
    )
```

**Pattern**: Pluggable discovery strategies for different network environments (UDP broadcast for local, manual for firewalled/cloud).

## Installation and Usage

### Quick Start

From [README.md](https://github.com/shamantechnology/xotorch/blob/main/README.md) (accessed 2025-11-02):

**Unix/Linux/macOS**:
```bash
# Install dependencies and setup
./install.sh

# Run XOTorch
xot
```

**Windows (PowerShell 7.5+)**:
```powershell
# Install
.\install.ps1

# Run
xot
```

**CUDA Users**: Ensure nvcc is installed to detect correct CUDA version for PyTorch installation.

### Command-Line Usage

**Run Model Inference**:
```bash
# Direct model execution
xot run llama-3.2-1b --prompt "Who are you?"

# Chat TUI mode
xot --chat-tui --default-model llama-3.2-3b
```

**Training Mode**:
```bash
xot train llama-3.2-1b \
    --data xotorch/train/data/lora \
    --batch-size 1 \
    --iters 100 \
    --save-every 5 \
    --save-checkpoint-dir checkpoints
```

**Evaluation**:
```bash
xot eval llama-3-8b --batch-size 2
```

### Discovery Configuration

**UDP Discovery (Default)**:
```bash
xot --discovery-module udp \
    --listen-port 5678 \
    --broadcast-port 5678 \
    --discovery-timeout 30
```

**Manual Discovery (Config-Based)**:
```bash
xot --discovery-module manual \
    --discovery-config-path config.json
```

**Node Filtering**:
```bash
# Filter by node IDs
xot --node-id-filter "node1,node2,node3"

# Filter by interface types
xot --interface-type-filter "ethernet,wifi"
```

## Key Patterns for TUI Development

### 1. Rich vs Textual Trade-offs

**XOTorch uses Rich, not Textual**:
- **Rich**: Simple live displays, static layouts, direct rendering
- **Textual**: Complex reactive UIs, event-driven widgets, CSS styling

**When to use Rich (XOTorch approach)**:
- Real-time monitoring dashboards
- Progress bars and downloads
- Simple chat interfaces
- ASCII art visualizations

**When to use Textual**:
- Interactive applications with forms
- Complex navigation and focus management
- Reactive state updates
- Multi-screen applications

### 2. Event-Driven Token Streaming

**Callback Registration Pattern**:
```python
# Register callback for specific request
callback = node.on_token.register(f"cli-wait-response-{request_id}")

# Process tokens as they arrive
def on_token(_request_id, _tokens, _is_finished):
    tokens.extend(_tokens)
    return _request_id == request_id and _is_finished

# Wait with timeout
await callback.wait(on_token, timeout=300)

# Always cleanup
node.on_token.deregister(callback_id)
```

**Pattern**: Request-scoped callbacks with guaranteed cleanup.

### 3. Distributed System Visualization

**Network Topology Layout**:
- Radial node positioning (circular arrangement)
- Color-coded status (active, current, peer)
- Dynamic panel visibility (only show active sections)
- Real-time metrics (FLOPS, memory, partitions)

### 4. Async/Await Integration

**Main Event Loop**:
```python
# Configure high-performance loop
loop = configure_uvloop()

# Run main async function
loop.run_until_complete(main())

# Cleanup
loop.close()
```

**Background Tasks**:
```python
# Non-blocking API server
asyncio.create_task(api.run(port=args.chatgpt_api_port))

# Preemptive model loading
asyncio.create_task(node.inference_engine.ensure_shard(current_shard))

# Progress broadcasting
asyncio.create_task(node.broadcast_opaque_status("", json.dumps(status)))
```

**Pattern**: Fire-and-forget tasks for background operations.

## Hardware Integration Notes

### CUDA and GPU Support

From [main.py](https://github.com/shamantechnology/xotorch/blob/main/xotorch/main.py) (accessed 2025-11-02):

**Inference Engine Selection**:
```python
# Lines 69-71: System detection
print(f"Selected inference engine: {args.inference_engine}")
system_info = get_system_info()
print(f"Detected system: {system_info}")
```

**Device Capabilities**:
- FLOPS measurement (FP16 performance)
- Memory detection (VRAM for GPUs)
- Model partitioning based on hardware

### Jetson and Edge Devices

From project description and fork rationale:

**PyTorch Focus Benefits**:
- Better cross-platform support (x86, ARM, Jetson)
- Wider hardware compatibility vs MLX/Metal-specific solutions
- Community ecosystem for optimization

**Distributed Inference**:
- Partition models across multiple devices
- Network-based shard coordination
- Automatic capability detection

## Development Resources

**Project Links**:
- GitHub: https://github.com/shamantechnology/xotorch
- Discord: https://discord.gg/j6bq3E44VR
- X/Twitter: https://x.com/shamantekllc
- Task Board: https://github.com/orgs/shamantechnology/projects/3

**Related Projects**:
- Original Exo: https://github.com/exo-explore/exo (v1 open source, v2 closed)
- PyTorch: https://pytorch.org
- Rich Library: https://rich.readthedocs.io

## Lessons for Textual Developers

### 1. Choose the Right Tool

**Rich is sufficient when**:
- You need real-time metrics display
- Simple chat/logging interface
- Progress bars and status indicators
- ASCII art visualizations

**Textual is better when**:
- Complex user interactions
- Forms and input validation
- Multi-screen navigation
- Reactive state management

### 2. Performance Matters

**XOTorch optimizations applicable to Textual**:
- uvloop/winloop for async performance
- Thread pool sizing based on CPU count
- File descriptor limit increases
- Rate-limited UI updates (200ms throttle)

### 3. Distributed System Patterns

**Event Broadcasting**:
- Throttled updates to prevent spam
- Callback registration/deregistration
- Request-scoped event handlers
- Background task management

### 4. ML Inference UI Challenges

**Token Streaming**:
- Decode tokens incrementally
- Display partial responses
- Calculate real-time throughput
- Handle backpressure

**Model Management**:
- Show download progress
- Display network topology
- Monitor distributed coordination
- Track hardware utilization

## Sources

**GitHub Repository**:
- [shamantechnology/xotorch](https://github.com/shamantechnology/xotorch) - Main repository (accessed 2025-11-02)
- [README.md](https://github.com/shamantechnology/xotorch/blob/main/README.md) - Project overview (accessed 2025-11-02)
- [main.py](https://github.com/shamantechnology/xotorch/blob/main/xotorch/main.py) - Application entry point (accessed 2025-11-02, 402 lines)
- [chat_tui.py](https://github.com/shamantechnology/xotorch/blob/main/xotorch/viz/chat_tui.py) - Chat interface (accessed 2025-11-02, 166 lines)
- [topology_viz.py](https://github.com/shamantechnology/xotorch/blob/main/xotorch/viz/topology_viz.py) - Network visualization (accessed 2025-11-02, 379 lines)

**Project Metadata**:
- License: GPL-3.0
- Stars: 28
- Forks: 5
- Contributors: 3
- Languages: Python 84.2%, JavaScript 6.9%, HTML 4.7%, CSS 3.6%
- Last updated: 2025-07-06 (README.md)
