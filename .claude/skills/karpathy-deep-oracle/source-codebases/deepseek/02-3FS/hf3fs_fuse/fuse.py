# <claudes_code_comments>
# ** Function List **
# get_mount_point(p) - Extract 3FS mount point from file path
#
# ** Technical Review **
# Simplified FUSE mount point extraction utility for hf3fs_fuse package.
#
# Function: get_mount_point(p)
# - Purpose: Extract the base mount point from any path within a 3FS mount
# - Algorithm:
#   1. Resolve symlinks: os.path.realpath(p)
#   2. Split into parts: PosixPath(np).parts
#   3. Take first 3 components: parts[:3]
#   4. Rejoin: os.path.join(*parts[:3])
#
# Expected Path Structure:
# - Input: Any path like '/hf3fs-cluster/mount_name/deep/nested/file.txt'
# - parts[:3] → ('/', 'hf3fs-cluster', 'mount_name')
# - Output: '/hf3fs-cluster/mount_name'
#
# Example:
# ```python
# path = '/hf3fs-cluster/gpu/data/checkpoints/model.pth'
# mount = get_mount_point(path)  # '/hf3fs-cluster/gpu'
# ```
#
# Use Cases:
# - Determine FUSE mount root for ioctl operations
# - Validate paths belong to same mount point
# - Extract mount name for client initialization
#
# Differences from hf3fs.fuse module:
# - hf3fs.fuse.mountName() → extracts just 'mount_name' string
# - hf3fs.fuse.serverPath() → extracts server-side path '/deep/nested/file.txt'
# - get_mount_point() → extracts full mount path '/hf3fs-cluster/mount_name'
#
# Integration:
# - Used by hf3fs_fuse.io for ioring initialization
# - Complements hf3fs.fuse path utilities
# - Simpler interface for mount point operations
# </claudes_code_comments>

import os
from pathlib import PosixPath

def get_mount_point(p):
    np = os.path.realpath(p)
    parts = PosixPath(np).parts
    return os.path.join(*parts[:3])
