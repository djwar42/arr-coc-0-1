# <claudes_code_comments>
# ** Function List **
# serverPath(p) - Extract server-side path from client mount path
# mountName(p) - Extract mount name from full path
#
# ** Technical Review **
# 3FS FUSE path utility module providing path parsing for client-server communication.
#
# Path Structure:
# - Full client path: /hf3fs-cluster/mount_name/server/path/to/file
# - Parts breakdown:
#   * parts[0]: '/' (root)
#   * parts[1]: 'hf3fs-cluster' (FUSE mount point base)
#   * parts[2]: mount_name (logical cluster/volume name)
#   * parts[3+]: server-side path components
#
# serverPath() Flow:
# 1. Normalize path: os.path.normpath(os.path.realpath(p))
#    - Resolves symlinks via realpath()
#    - Removes '..' and '.' via normpath()
# 2. Extract parts: PosixPath(np).parts
# 3. Reconstruct server path: '/' + parts[3:]
# 4. Example: '/hf3fs-cluster/cpu/../gpu/data/file.txt'
#    → realpath → '/hf3fs-cluster/gpu/data/file.txt'
#    → parts → ('/', 'hf3fs-cluster', 'gpu', 'data', 'file.txt')
#    → server path → '/data/file.txt'
#
# mountName() Flow:
# 1. Same normalization as serverPath()
# 2. Extract parts[2] directly (mount name)
# 3. Example: '/hf3fs-cluster/cpu/../gpu/data/file.txt'
#    → realpath → '/hf3fs-cluster/gpu/data/file.txt'
#    → parts → ('/', 'hf3fs-cluster', 'gpu', 'data', 'file.txt')
#    → mount name → 'gpu'
#
# IOctl Constants:
# - HF3FS_SUPER_MAGIC: Magic number for 3FS filesystem identification
# - HF3FS_IOC_GET_MOUNT_NAME (2149607424): Retrieve mount name via ioctl
# - HF3FS_IOC_GET_PATH_OFFSET (2147772417): Get path offset in mount structure
# - HF3FS_IOC_GET_MAGIC_NUM (2147772418): Verify filesystem magic number
# - HF3FS_IOC_RECURSIVE_RM (2147772426): Recursive remove operation (trash support)
#
# These constants are ioctl command codes for 3FS-specific operations:
# - Mount name retrieval: Used to identify which cluster a path belongs to
# - Path offset: Used for internal FUSE path resolution optimization
# - Magic number: Filesystem type verification before operations
# - Recursive rm: Efficient directory deletion with trash support (move to .trash/)
#
# Usage Pattern:
# ```python
# import hf3fs.fuse
# full_path = '/hf3fs-cluster/gpu/data/model.pth'
# server_path = hf3fs.fuse.serverPath(full_path)  # '/data/model.pth'
# mount = hf3fs.fuse.mountName(full_path)  # 'gpu'
# ```
#
# Integration:
# - Used by hf3fs.Client for path translation before RPC calls
# - FUSE mount structure assumes /hf3fs-cluster as base
# - Mount names map to different storage clusters or volumes
# - Server paths are relative to mount root (cluster namespace)
# </claudes_code_comments>

import os
from pathlib import PosixPath

from hf3fs_py_usrbio import HF3FS_SUPER_MAGIC

HF3FS_IOC_GET_MOUNT_NAME = 2149607424
HF3FS_IOC_GET_PATH_OFFSET = 2147772417
HF3FS_IOC_GET_MAGIC_NUM = 2147772418

HF3FS_IOC_RECURSIVE_RM = 2147772426

def serverPath(p):
    '''
    从完整路径获取 client 接受的路径名

    Args:
        p: 待解析的路径名

    Examples:

    .. code-block:: python

        import hf3fs.fuse
        hf3fs.fuse.serverPath('/hf3fs-cluster/aaa/../cpu/abc/def')
    '''
    np = os.path.normpath(os.path.realpath(p))
    return os.path.join('/', *PosixPath(np).parts[3:])

def mountName(p):
    '''
    从完整路径获取 mount name

    Args:
        p: 待解析的路径名

    Examples:
    
    .. code-block:: python
    
        import hf3fs.fuse
        hf3fs.fuse.mountName('/hf3fs-cluster/aaa/../cpu/abc/def')
    '''
    np = os.path.normpath(os.path.realpath(p))
    return PosixPath(np).parts[2]
