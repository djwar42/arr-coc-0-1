# terraform-tui - Infrastructure State Viewer

**Source**: [GitHub - idoavrah/terraform-tui](https://github.com/idoavrah/terraform-tui)
**Accessed**: 2025-11-02
**Repository**: https://github.com/idoavrah/terraform-tui
**PyPI**: https://pypi.org/project/tftui/
**License**: Apache-2.0
**Stars**: 1.2k | **Forks**: 39

## Overview

TFTUI (Terraform Textual UI) is a powerful Textual-based TUI that empowers users to view and interact with Terraform infrastructure state. It provides comprehensive visualization of the complete state tree, resource inspection, and direct infrastructure manipulation capabilities.

**Key Innovation**: Real-time Terraform state interaction - view, search, modify, plan, and apply infrastructure changes directly from the terminal interface.

## Core Features

### State Visualization
- **Complete State Tree Display**: Hierarchical view of entire Terraform state
- **Resource Navigation**: Browse and inspect individual resource states
- **Search Functionality**: Search through state tree and resource definitions
- **Module Support**: Navigate through full module names and structures
- **Sensitive Values Extraction**: Display sensitive values from resource details (v0.13)

### Infrastructure Operations
- **Plan Creation**: Create Terraform plans with full color formatting
- **Plan Application**: Apply plans directly from the TUI
- **Targeted Plans**: Target specific resources for plan creation (v0.12)
- **Resource Selection**: Single and multiple resource selection
- **Resource Operations**:
  - Taint resources
  - Untaint resources
  - Delete resources
  - Destroy resources

### Workspace Management
- **Workspace Switching**: Switch between Terraform workspaces (v0.13)
- **Empty State Handling**: Show empty tree when no state exists, allowing plan creation (v0.13)
- **Plan Summary**: Display plan summary in screen title (v0.13)

### User Experience
- **Vim-like Navigation**: Support for vim keybindings (v0.13)
- **Fullscreen Mode**: Easier copying of resource/plan parts (v0.12)
- **Help Screen**: Built-in help documentation (v0.12)
- **Confirmation Dialogs**: Modal screens for critical operations (v0.11)
- **Visual Indicators**: Color-coded tainted resources
- **Dynamic UI**: Targets checkbox reflects resource selection state

## External Tool Integration

### Terraform Integration Pattern

TFTUI demonstrates **direct CLI integration** with Terraform:

**Architecture**:
```
Textual TUI (TFTUI)
    ↓
Execute Terraform CLI Commands
    ↓
Parse JSON Output (terraform state/plan)
    ↓
Present in Textual Widgets
    ↓
User Actions → More Terraform Commands
```

**Key Integration Points**:

1. **State Reading**: Execute `terraform state` commands to fetch infrastructure state
2. **State Modification**: Execute `terraform taint/untaint/destroy` for resource operations
3. **Plan Operations**: Execute `terraform plan` with optional targets and tfvars
4. **Apply Operations**: Execute `terraform apply` to realize infrastructure changes
5. **Workspace Operations**: Execute `terraform workspace` commands for switching

### Wrapper Support

**Terragrunt Compatibility**: Supports Terraform wrappers like Terragrunt, allowing use with infrastructure-as-code orchestration tools.

**Pattern**: The tool adapts to use wrapper commands instead of direct Terraform CLI, demonstrating flexible external tool integration.

## Installation

### Multiple Installation Methods

```bash
# Homebrew (macOS/Linux)
brew install idoavrah/homebrew/tftui
brew upgrade tftui

# PIP
pip install tftui
pip install --upgrade tftui

# PIPX (isolated environment)
pipx install tftui
pipx upgrade tftui
```

### Usage

```bash
# Navigate to Terraform project directory
cd /path/to/terraform/project

# Run TFTUI
tftui

# Offline mode (no outbound API calls)
tftui -o

# Disable usage tracking
tftui -d

# Specify tfvars file
tftui --tfvars=path/to/vars.tfvars
```

## Architecture Insights

### State Management Pattern

**Read-Execute-Present Cycle**:
1. **Read**: Execute Terraform CLI to fetch current state JSON
2. **Parse**: Convert JSON to internal data structures
3. **Present**: Render in Textual tree/list widgets
4. **Modify**: Execute Terraform commands based on user actions
5. **Refresh**: Re-read state and update display

### UI Components (Inferred)

Likely uses these Textual widgets:
- **Tree widget**: Display hierarchical state structure
- **Static/RichLog**: Display resource details and plan output
- **Input**: Search functionality
- **Button/Checkbox**: Resource selection and operations
- **Modal screens**: Confirmation dialogs and help screens
- **Header/Footer**: Plan summary and status information

### Privacy Considerations

**Usage Tracking**:
- Uses PostHog for anonymous usage analytics
- Fingerprint-based user identification (no personal data)
- Opt-out available via `-d` flag
- Offline mode (`-o`) disables all outbound API calls (v0.13)
- Default: No outbound calls when tracking disabled (v0.13)

## Version History (Recent)

### v0.13 (Latest)
- Workspace switching support
- Plan summary in screen title
- Empty state handling for new infrastructures
- Offline mode flag (`-o`)
- Removed default PostHog call when tracking disabled
- Sensitive values extraction
- Vim-like navigation support

### v0.12
- Targeted resource planning
- CLI tfvars file argument
- Destroy functionality
- Help screen
- Plan summary before apply
- Error tracking for unhandled exceptions
- Fullscreen mode
- Search through full module names
- Clipboard copy fixes

### v0.11
- Plan creation and application
- Modal confirmation dialogs
- Tainted resource coloring
- Improved loading screen

## Use Cases

### DevOps Workflows
- **State Inspection**: Quick terminal-based state review without switching to web UIs
- **Resource Management**: Taint/untaint resources for targeted updates
- **Plan Review**: Create and review plans before applying changes
- **Workspace Operations**: Switch between environments (dev/staging/prod)

### SRE Tasks
- **Incident Response**: Quickly identify and destroy problematic resources
- **State Verification**: Verify infrastructure matches expected configuration
- **Selective Updates**: Target specific resources for updates without full plan

### Development
- **Local Testing**: Test infrastructure changes in isolated workspaces
- **State Debugging**: Inspect resource attributes and dependencies
- **Rapid Iteration**: Create → review → apply cycle without leaving terminal

## Integration Pattern Analysis

**Textual + External CLI Tool Pattern**:

TFTUI demonstrates how to build a TUI that:
1. **Executes external commands** (Terraform CLI)
2. **Parses structured output** (JSON state/plan)
3. **Presents in interactive widgets** (tree, lists, modals)
4. **Translates UI actions to CLI commands** (taint → `terraform taint resource`)

**Key Lessons**:
- External tools don't need APIs - CLI + stdout parsing works
- JSON output from tools enables rich UI construction
- Textual handles complex interactions while tool does heavy lifting
- Wrapper support shows abstraction layer benefits

## Community & Adoption

**Metrics** (as of 2025-11-02):
- **Stars**: 1.2k (indicates strong community interest)
- **Forks**: 39 (active development community)
- **Contributors**: 4
- **Releases**: 35 versions (active maintenance)
- **Latest Release**: v0.13.5 (June 17, 2024)

**Language Distribution**:
- Python: 88.4% (Textual framework)
- HCL: 11.3% (Terraform test configurations)
- Shell: 0.3% (Build scripts)

## Related Projects

**Similar External Tool TUI Patterns**:
- K9s (Kubernetes cluster management TUI)
- Lazygit (Git operations TUI)
- Lazydocker (Docker management TUI)

**Pattern**: Terminal UIs for complex CLI tools where manual command execution is tedious.

## Sources

**GitHub Repository**:
- Main README: https://github.com/idoavrah/terraform-tui/blob/main/README.md
- Repository root: https://github.com/idoavrah/terraform-tui

**Additional Links**:
- PyPI package: https://pypi.org/project/tftui/
- Demo GIF: https://github.com/idoavrah/terraform-tui/blob/main/demo/tftui.gif
- Star History: https://star-history.com/#idoavrah/terraform-tui&Date
