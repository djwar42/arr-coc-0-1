"""DualPipe Package - Bidirectional Pipeline Parallelism for DeepSeek-V3

Exports:
- DualPipe: Bidirectional pipeline with 2x parameters per device, PP ranks
- DualPipeV: V-shape pipeline with 2x parameters per device, PP/2 devices
- set_p2p_tensor_shapes: Configure tensor shapes for communication
- set_p2p_tensor_dtype: Configure tensor dtype for communication
- WeightGradStore: Zero-bubble optimization utility

See individual modules for detailed documentation.
"""

__version__ = "1.0.0"

from dualpipe.dualpipe import DualPipe
from dualpipe.dualpipev import DualPipeV
from dualpipe.comm import (
    set_p2p_tensor_shapes,
    set_p2p_tensor_dtype,
)
from dualpipe.utils import WeightGradStore

__all__ = [
    DualPipe,
    DualPipeV,
    WeightGradStore,
    set_p2p_tensor_shapes,
    set_p2p_tensor_dtype,
]
