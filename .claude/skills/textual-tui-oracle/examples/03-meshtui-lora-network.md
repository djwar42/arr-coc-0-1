# MeshTUI - LoRa Network Management TUI

## Overview

**MeshTUI** is a cross-platform terminal user interface for Meshtastic LoRa mesh networks. Built with Python's `prompt_toolkit`, it provides real-time chat, node monitoring, and network management capabilities for off-grid LoRa communication devices.

**Key Characteristics:**
- Real-time hardware integration via serial ports
- Asynchronous message handling with pub/sub patterns
- Dynamic device discovery and switching
- Network topology visualization
- Cross-platform (Linux, macOS, Windows with .exe)

**Repository**: https://github.com/SAMS0N1TE/meshtui
**License**: AGPL-3.0
**Stars**: 24+ (as of 2025-11-02)

---

## Technical Stack

### Core Dependencies

From [requirements.txt](https://github.com/SAMS0N1TE/meshtui/blob/main/requirements.txt):

```text
meshtastic        # Meshtastic Python API for LoRa devices
pyserial          # Serial port communication
prompt-toolkit    # TUI framework (alternative to Textual)
pypubsub          # Publisher-subscriber pattern for events
```

**Why prompt_toolkit instead of Textual?**
- MeshTUI uses `prompt_toolkit` (older TUI framework)
- More low-level control over terminal rendering
- Lighter weight for simple chat interfaces
- Direct curses-like event handling

**Note**: While not built with Textual, MeshTUI demonstrates valuable patterns applicable to Textual-based LoRa/hardware projects.

---

## Architecture Patterns

### 1. Hardware Integration via Serial Ports

**Dynamic Serial Port Detection:**
```python
# Pattern: Auto-detect available serial ports
import serial.tools.list_ports

def detect_ports():
    """Scan for available serial devices"""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

# Runtime port switching via TUI menu
# User can switch between multiple connected LoRa devices
```

**Serial Communication Flow:**
```
User Input (TUI) → Command Queue → Serial Port → LoRa Radio
LoRa Radio → Serial Port → Event Parser → UI Update
```

### 2. Real-Time Message Handling

**Pub/Sub Pattern with pypubsub:**
```python
# Pattern: Decouple hardware events from UI updates
from pubsub import pub

# Publisher (hardware layer)
def on_receive_message(packet):
    """Called when LoRa radio receives data"""
    pub.sendMessage('mesh.message.received', packet=packet)

# Subscriber (UI layer)
def update_chat_display(packet):
    """Update TUI chat window"""
    message = parse_packet(packet)
    chat_window.add_message(message)

pub.subscribe(update_chat_display, 'mesh.message.received')
```

**Asynchronous Message Processing:**
- Background thread monitors serial port
- Messages queued for UI thread
- Non-blocking UI updates
- Message status tracking (sent/delivered/error)

### 3. Node Discovery and Monitoring

**Node List Management:**
```python
# Pattern: Maintain dynamic node registry
class NodeRegistry:
    def __init__(self):
        self.nodes = {}  # {node_id: NodeInfo}

    def update_node(self, node_id, snr, last_heard):
        """Update node metadata"""
        self.nodes[node_id] = NodeInfo(
            id=node_id,
            snr=snr,  # Signal-to-noise ratio
            last_heard=time.time(),
            status='active' if recent else 'stale'
        )

    def get_active_nodes(self):
        """Filter nodes heard recently"""
        cutoff = time.time() - 900  # 15 minutes
        return [n for n in self.nodes.values()
                if n.last_heard > cutoff]
```

**Node Metadata Displayed:**
- Node ID/Name
- Signal strength (SNR)
- Last heard timestamp
- GPS coordinates (if available)
- Battery status

### 4. Network Visualization

**Map Display Features:**
- ASCII/Unicode map rendering in terminal
- Node positions from GPS data
- Connection lines between nodes
- Signal strength color coding

**Map Update Pattern:**
```python
# Pattern: Incremental map updates
class MapWidget:
    def __init__(self, width, height):
        self.grid = [[' ' for _ in range(width)]
                     for _ in range(height)]
        self.nodes = []

    def add_node(self, lat, lon, node_id):
        """Convert GPS to terminal coordinates"""
        x, y = self.lat_lon_to_grid(lat, lon)
        self.grid[y][x] = node_id[0]  # First letter

    def render(self):
        """Draw map to terminal"""
        return '\n'.join(''.join(row) for row in self.grid)
```

---

## Use Case: Off-Grid Messaging

**Scenario**: Remote communication without cellular/internet

**Network Topology:**
```
[Node A] ←→ [Node B] ←→ [Node C]
   ↑                       ↑
   └──────── [Node D] ─────┘

LoRa Mesh: Messages hop between nodes
Range: 1-10km per hop (terrain dependent)
```

**TUI Workflow:**
1. **Connect**: Select serial port from detected devices
2. **Monitor**: View active nodes and signal strength
3. **Chat**: Send broadcast or direct messages
4. **Route**: Traceroute to specific nodes
5. **Map**: Visualize network topology

---

## Key TUI Patterns for Hardware Projects

### Pattern 1: Serial Port Selection Menu

**Challenge**: Multiple devices, runtime switching

**Solution:**
```python
# Dynamic menu populated from hardware detection
def build_port_menu():
    ports = detect_ports()
    menu_items = [
        MenuItem(f"Serial Port: {port}",
                 callback=lambda p=port: connect(p))
        for port in ports
    ]
    menu_items.append(MenuItem("Refresh Ports",
                               callback=build_port_menu))
    return menu_items
```

**Applicable to Textual:**
```python
from textual.widgets import Select

class PortSelector(Select):
    def on_mount(self):
        self.populate_ports()

    def populate_ports(self):
        ports = detect_ports()
        self.set_options([(p, p) for p in ports])

    def on_select_changed(self, event):
        connect_to_port(event.value)
```

### Pattern 2: Real-Time Status Display

**Challenge**: Show live hardware metrics (SNR, battery, etc.)

**Solution with prompt_toolkit:**
```python
# Live-updating status bar
class StatusBar:
    def get_text(self):
        return f"SNR: {current_snr} | Battery: {battery}% | Nodes: {node_count}"
```

**Textual Equivalent:**
```python
from textual.reactive import reactive
from textual.widgets import Footer

class HardwareFooter(Footer):
    snr = reactive(0)
    battery = reactive(100)
    node_count = reactive(0)

    def render(self):
        return f"SNR: {self.snr} | Battery: {self.battery}% | Nodes: {self.node_count}"

    # Update from background thread
    def update_metrics(self, snr, battery, nodes):
        self.snr = snr
        self.battery = battery
        self.node_count = nodes
```

### Pattern 3: Message Queue with Status Tracking

**Challenge**: Track message delivery over unreliable network

**Message States:**
```python
from enum import Enum

class MessageStatus(Enum):
    QUEUED = "Queued"
    SENDING = "Sending..."
    SENT = "Sent"
    DELIVERED = "✓ Delivered"
    ERROR = "✗ Failed"
```

**Status Display:**
```python
# In chat window, show message with status icon
def render_message(msg):
    status_icon = {
        MessageStatus.QUEUED: '◌',
        MessageStatus.SENDING: '○',
        MessageStatus.SENT: '◉',
        MessageStatus.DELIVERED: '✓',
        MessageStatus.ERROR: '✗'
    }[msg.status]

    return f"{msg.timestamp} [{status_icon}] {msg.sender}: {msg.text}"
```

### Pattern 4: Background Hardware Monitoring

**Challenge**: Non-blocking serial port monitoring

**Threading Pattern:**
```python
import threading
from queue import Queue

class HardwareMonitor:
    def __init__(self):
        self.message_queue = Queue()
        self.running = False

    def start(self):
        """Start background monitoring thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def _monitor_loop(self):
        """Background thread: read serial port"""
        while self.running:
            if serial_port.in_waiting:
                data = serial_port.read_until()
                packet = parse_packet(data)
                self.message_queue.put(packet)
            time.sleep(0.01)  # Don't burn CPU

    def process_queue(self):
        """UI thread: process queued messages"""
        while not self.message_queue.empty():
            packet = self.message_queue.get()
            self.handle_packet(packet)
```

**Textual Integration:**
```python
from textual.worker import Worker

class LoRaMonitor(App):
    def on_mount(self):
        self.run_worker(self.monitor_serial_port())

    async def monitor_serial_port(self) -> None:
        """Background worker for serial monitoring"""
        while True:
            if serial_port.in_waiting:
                data = await asyncio.to_thread(serial_port.read_until)
                packet = parse_packet(data)
                self.post_message(LoRaMessage(packet))
            await asyncio.sleep(0.01)
```

---

## Message Types and Routing

### Broadcast Messages
```python
# Send to all nodes on mesh
def send_broadcast(text):
    packet = create_packet(
        to=BROADCAST_ID,  # 0xFFFFFFFF
        text=text,
        want_ack=False
    )
    send_packet(packet)
```

### Direct Messages (DM)
```python
# Send to specific node
def send_dm(node_id, text):
    packet = create_packet(
        to=node_id,
        text=text,
        want_ack=True  # Request delivery confirmation
    )
    send_packet(packet)
    track_message_status(packet.id)
```

### Traceroute
```python
# Discover route to node
def traceroute(target_node_id):
    packet = create_traceroute_request(target_node_id)
    send_packet(packet)
    # Response shows hops: A → B → C → Target
```

---

## Real-Time Data Considerations

### 1. Serial Port Buffering

**Challenge**: Data arrives in chunks, not complete packets

**Solution:**
```python
class PacketBuffer:
    def __init__(self):
        self.buffer = bytearray()

    def add_data(self, data):
        """Accumulate incoming bytes"""
        self.buffer.extend(data)

    def extract_packets(self):
        """Parse complete packets from buffer"""
        packets = []
        while len(self.buffer) >= MIN_PACKET_SIZE:
            # Find packet delimiter
            if packet_complete(self.buffer):
                packet_bytes = self.buffer[:packet_size]
                packets.append(parse(packet_bytes))
                self.buffer = self.buffer[packet_size:]
            else:
                break  # Wait for more data
        return packets
```

### 2. Rate Limiting

**Challenge**: Don't flood LoRa network (limited bandwidth)

**Solution:**
```python
from time import time

class RateLimiter:
    def __init__(self, max_per_minute=20):
        self.max_per_minute = max_per_minute
        self.timestamps = []

    def can_send(self):
        """Check if under rate limit"""
        now = time()
        # Remove old timestamps
        self.timestamps = [t for t in self.timestamps
                          if now - t < 60]
        return len(self.timestamps) < self.max_per_minute

    def record_send(self):
        """Track message sent"""
        self.timestamps.append(time())
```

### 3. Stale Node Detection

**Challenge**: Nodes go offline without notification

**Solution:**
```python
# Pattern: Age-out inactive nodes
STALE_THRESHOLD = 900  # 15 minutes

def update_node_status():
    """Called periodically"""
    now = time.time()
    for node in node_registry.values():
        if now - node.last_heard > STALE_THRESHOLD:
            node.status = 'stale'
            ui.mark_node_inactive(node.id)
```

---

## Hardware Integration Considerations

### 1. Device Compatibility

**Supported LoRa Devices:**
- Heltec LoRa32 V3
- LilyGo T-Deck
- T-Beam
- Any device running Meshtastic firmware

**Connection Methods:**
- USB serial (/dev/ttyUSB*, /dev/ttyACM*, COM*)
- Bluetooth (via meshtastic Python API)
- Network (TCP/IP to Meshtastic device)

### 2. Error Handling

**Serial Port Disconnection:**
```python
try:
    data = serial_port.read()
except serial.SerialException:
    # Device unplugged or connection lost
    show_error("LoRa device disconnected")
    revert_to_port_selection()
```

**Timeout Handling:**
```python
# Don't wait forever for acknowledgments
def send_with_timeout(packet, timeout=30):
    ack_event = threading.Event()

    def on_ack(ack_packet):
        if ack_packet.id == packet.id:
            ack_event.set()

    pub.subscribe(on_ack, 'mesh.ack.received')
    send_packet(packet)

    if ack_event.wait(timeout):
        return MessageStatus.DELIVERED
    else:
        return MessageStatus.ERROR
```

### 3. Power Management

**Battery Monitoring:**
```python
# Display battery level for remote nodes
class NodeInfo:
    battery_level: int  # 0-100%
    is_charging: bool
    voltage: float

def render_battery_icon(level):
    """Visual battery indicator"""
    if level > 75: return '█████'
    if level > 50: return '████░'
    if level > 25: return '███░░'
    if level > 10: return '██░░░'
    return '█░░░░'  # Low battery warning
```

---

## Cross-Platform Deployment

### Linux/macOS (From Source)

```bash
# Clone repository
git clone https://github.com/SAMS0N1TE/meshtui.git
cd meshtui

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python start_meshtui.py
```

### Windows (Executable)

```
Download: MeshTUI.exe from GitHub releases
Double-click to run (no Python installation needed)
```

**Build Process (PyInstaller):**
```bash
# Create standalone .exe
pip install pyinstaller
pyinstaller --onefile --windowed start_meshtui.py
# Output: dist/MeshTUI.exe
```

---

## UI Layout Structure

### Main Screen Layout

```
╔════════════════════════════════════════════════════════════╗
║ MeshTUI - Connected to /dev/ttyUSB0                       ║
╠════════════════════════════════════════════════════════════╣
║ Node List (Left Panel)     │ Chat/Map (Right Panel)       ║
║                             │                              ║
║ ○ HomeBase (SNR: 12)        │ [09:45] Alice: Hello mesh!  ║
║ ○ Rover-1 (SNR: 8)          │ [09:46] Bob: Copy that      ║
║ ◌ Remote-2 (SNR: -2) stale  │ [09:47] You: Testing 123    ║
║ ○ Mobile-3 (SNR: 15)        │                              ║
║                             │ [Map View]                   ║
║ [T] Traceroute              │    A ←→ B                    ║
║ [M] Message                 │     ↖  ↓                     ║
║ [V] Toggle Map              │       C                      ║
║                             │                              ║
╠════════════════════════════════════════════════════════════╣
║ SNR: 12 | Battery: 85% | Nodes: 3 | Messages: 24         ║
╚════════════════════════════════════════════════════════════╝
```

**Layout Components:**
1. **Title Bar**: Connection status, port name
2. **Left Panel**: Active node list with metrics
3. **Right Panel**: Chat history or map view (toggle)
4. **Status Bar**: Real-time hardware metrics
5. **Input Area**: Message composition (not shown, bottom)

---

## Keybindings

**Navigation:**
- `Tab` - Switch focus between panels
- `↑/↓` - Navigate node list
- `PgUp/PgDn` - Scroll chat history

**Actions:**
- `Enter` - Send message / Select node
- `Ctrl+M` - Direct message to selected node
- `Ctrl+T` - Traceroute to selected node
- `Ctrl+V` - Toggle chat/map view
- `Ctrl+P` - Change serial port
- `Ctrl+Q` - Quit application

**Message Status:**
- `◌` - Queued
- `○` - Sending
- `◉` - Sent
- `✓` - Delivered
- `✗` - Failed

---

## Lessons for Textual-Based Hardware Projects

### 1. Use Workers for Hardware I/O

**Don't block UI thread with serial reads:**
```python
from textual.worker import Worker

class HardwareTUI(App):
    @work(exclusive=True, thread=True)
    async def monitor_hardware(self):
        """Background worker for blocking I/O"""
        while True:
            data = serial_port.read()  # Blocking call
            self.post_message(HardwareData(data))
```

### 2. Reactive Properties for Live Metrics

**Automatically update UI when hardware state changes:**
```python
from textual.reactive import reactive

class NodeListWidget(Widget):
    nodes = reactive({})  # Auto-refresh when updated

    def watch_nodes(self, old_nodes, new_nodes):
        """Called automatically when nodes changes"""
        self.refresh()  # Redraw widget
```

### 3. Message Passing for Decoupling

**Use Textual's message system:**
```python
# Hardware event as message
class LoRaPacketReceived(Message):
    def __init__(self, packet):
        self.packet = packet
        super().__init__()

# UI widget handles it
class ChatPanel(Widget):
    def on_lo_ra_packet_received(self, message):
        """Automatically called when message posted"""
        self.add_message(message.packet)
```

### 4. Command Palette for Complex Actions

**MeshTUI uses keybindings, but Textual CommandPalette is more discoverable:**
```python
from textual.command import Provider

class LoRaCommands(Provider):
    async def search(self, query: str):
        """Searchable commands"""
        yield Command("Send Broadcast", self.send_broadcast)
        yield Command("Traceroute to Node", self.traceroute)
        yield Command("Change Serial Port", self.select_port)
```

### 5. Validate Input Before Hardware Send

**Prevent invalid data from reaching hardware:**
```python
from textual.validation import Validator

class MessageValidator(Validator):
    def validate(self, value: str) -> bool:
        # LoRa has strict size limits
        return len(value.encode('utf-8')) <= 200

    def get_error(self) -> str:
        return "Message too long (max 200 bytes)"
```

---

## Performance Optimization

### 1. Lazy Rendering

**Don't redraw entire UI for every packet:**
```python
# Only update affected widgets
def on_new_message(message):
    chat_panel.append_message(message)  # Incremental
    # Don't: rebuild entire UI
```

### 2. Message Batching

**Group UI updates when burst of packets arrive:**
```python
class MessageBatcher:
    def __init__(self, delay=0.1):
        self.pending = []
        self.timer = None

    def add_message(self, msg):
        self.pending.append(msg)
        if self.timer is None:
            self.timer = threading.Timer(0.1, self.flush)
            self.timer.start()

    def flush(self):
        """Update UI once with all messages"""
        ui.add_messages(self.pending)
        self.pending = []
        self.timer = None
```

### 3. Limited History

**Don't store unlimited messages in memory:**
```python
class ChatHistory:
    MAX_MESSAGES = 1000

    def add_message(self, msg):
        self.messages.append(msg)
        if len(self.messages) > self.MAX_MESSAGES:
            self.messages = self.messages[-self.MAX_MESSAGES:]
```

---

## Testing Hardware TUIs

### Mock Serial Port

**Test without physical hardware:**
```python
class MockSerialPort:
    def __init__(self):
        self.write_buffer = []
        self.read_buffer = []

    def write(self, data):
        """Simulate sending to hardware"""
        self.write_buffer.append(data)
        # Auto-reply with ACK
        self.read_buffer.append(create_ack(data))

    def read(self):
        """Simulate receiving from hardware"""
        if self.read_buffer:
            return self.read_buffer.pop(0)
        return b''
```

### Simulated Node Network

**Test with fake mesh network:**
```python
class SimulatedMesh:
    def __init__(self):
        self.nodes = {
            'NODE_A': NodeInfo(snr=12, battery=85),
            'NODE_B': NodeInfo(snr=8, battery=60),
            'NODE_C': NodeInfo(snr=-2, battery=40)
        }

    def send_message(self, to, text):
        """Simulate message propagation"""
        if to in self.nodes:
            # Random delay (0.5-2s)
            delay = random.uniform(0.5, 2.0)
            threading.Timer(delay, self._deliver, [to, text]).start()

    def _deliver(self, to, text):
        """Simulate delivery with random success"""
        if random.random() < 0.9:  # 90% success rate
            pub.sendMessage('mesh.ack.received', node=to)
        else:
            pub.sendMessage('mesh.timeout', node=to)
```

---

## Community and Resources

**Project Links:**
- **GitHub Repository**: https://github.com/SAMS0N1TE/meshtui
- **Reddit Discussion**: https://www.reddit.com/r/meshtastic/comments/1l82dan/meshtui_a_simple_tui_for_sending_messages/
- **Meshtastic Project**: https://meshtastic.org/
- **Meshtastic Python API**: https://github.com/meshtastic/python

**Related Subreddits:**
- r/meshtastic - Meshtastic community
- r/LoRa - General LoRa technology
- r/LoRaWAN - LoRaWAN protocol discussions

**Hardware Vendors:**
- Heltec Automation (LoRa32 boards)
- LilyGo (T-Deck, T-Beam devices)

---

## Comparison: prompt_toolkit vs Textual

### Why MeshTUI Uses prompt_toolkit

**prompt_toolkit Advantages:**
- Lower-level control over rendering
- Smaller memory footprint
- Simpler event loop for basic UIs
- Mature, stable API

**When to Choose Textual Instead:**
- Complex layouts with grids/docks
- Need reactive data binding
- Want built-in widgets (DataTable, Tree, etc.)
- Prefer declarative CSS styling
- Need browser deployment (textual-web)

### Porting MeshTUI to Textual

**Would benefit from:**
- `DataTable` for node list (sortable columns)
- `TabbedContent` for chat/map/settings views
- CSS styling for consistent theme
- `CommandPalette` for discoverability
- Reactive properties for live metrics

**Example Textual Port:**
```python
from textual.app import App
from textual.widgets import DataTable, TextLog, Footer
from textual.reactive import reactive

class MeshTUITextual(App):
    CSS = """
    DataTable { dock: left; width: 30%; }
    TextLog { dock: right; }
    """

    nodes = reactive({})  # Auto-refresh table

    def compose(self):
        yield DataTable()
        yield TextLog()
        yield Footer()

    def on_mount(self):
        self.query_one(DataTable).add_columns("Node", "SNR", "Battery")
        self.run_worker(self.monitor_lora())

    @work(exclusive=True)
    async def monitor_lora(self):
        """Background LoRa monitoring"""
        while True:
            packet = await read_serial_async()
            self.post_message(LoRaPacket(packet))
```

---

## Key Takeaways

### For Hardware Integration
1. **Use background workers** for blocking I/O (serial ports, hardware reads)
2. **Implement pub/sub** to decouple hardware from UI
3. **Queue messages** for batch processing in UI thread
4. **Handle disconnections** gracefully with error recovery
5. **Rate limit** to prevent hardware overload

### For Real-Time Data
1. **Age-out stale data** (nodes not heard recently)
2. **Buffer incomplete packets** until full message received
3. **Track message status** through delivery lifecycle
4. **Batch UI updates** when data arrives in bursts
5. **Limit history size** to prevent memory bloat

### For Network Visualization
1. **Use ASCII/Unicode** for terminal-native graphics
2. **Convert GPS to grid coordinates** for mapping
3. **Color-code metrics** (SNR, battery) for quick scanning
4. **Update incrementally** (don't redraw full map)
5. **Provide text fallback** when map unavailable

---

## Sources

**Primary Sources:**
- [MeshTUI GitHub Repository](https://github.com/SAMS0N1TE/meshtui) - Main codebase (accessed 2025-11-02)
- [MeshTUI README](https://github.com/SAMS0N1TE/meshtui/blob/main/README.md) - Installation and features
- [MeshTUI requirements.txt](https://github.com/SAMS0N1TE/meshtui/blob/main/requirements.txt) - Dependencies

**Community Sources:**
- [Reddit r/meshtastic Discussion](https://www.reddit.com/r/meshtastic/comments/1l82dan/meshtui_a_simple_tui_for_sending_messages/) - User feedback and use cases (accessed 2025-11-02)

**Related Documentation:**
- Meshtastic Project: https://meshtastic.org/
- prompt_toolkit Documentation: https://python-prompt-toolkit.readthedocs.io/
- PySerial Documentation: https://pyserial.readthedocs.io/

---

**Document Created**: 2025-11-02
**Oracle**: textual-tui-oracle
**Type**: Hardware Integration Example
**Category**: Real-Time Communication TUI
