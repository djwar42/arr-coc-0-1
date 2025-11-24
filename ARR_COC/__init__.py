"""
ARR-COC 0.1 - Adaptive Relevance Realization for Vision-Language Models

A minimal viable implementation of Vervaekean relevance realization.
"""

__version__ = "0.1.0"

from .texture import generate_texture_array
from .knowing import information_score, perspectival_score, ParticipatoryScorer
from .balancing import AdaptiveTensionBalancer
from .attending import TokenAllocator
from .integration import ARRCOCQwen  # NOW AVAILABLE!

__all__ = [
    "generate_texture_array",
    "information_score",
    "perspectival_score",
    "ParticipatoryScorer",
    "AdaptiveTensionBalancer",
    "TokenAllocator",
    "ARRCOCQwen",  # Main entry point
]
