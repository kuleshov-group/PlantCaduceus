"""Experiment setting choices.

Includes:
    - AttentionStrategy
    - FineTuningStrategy
    - PositionalEmbeddingStrategy
"""
from enum import Enum


class AttentionStrategy(Enum):
    """Choices for attention strategy."""
    ORIGINAL = "original"
    XFORMER = "xformer"
    XFORMER_LOCAL = "xformer_local"
    XFORMER_EFFICIENT = "xformer_efficient"
    # XFORMER_RANDOM = "xformer_random"
    # XFORMER_LINFORMER = "xformer_linformer"


class FineTuningStrategy(Enum):
    """Choices for fine-tuning strategy."""
    FULL = "full"
    LORA = "lora"


class PositionalEmbeddingStrategy(Enum):
    """Choices for positional embedding interpolation strategy."""
    INTERPOLATE = "interpolate"
    NONE = "none"
    NTK = "ntk"
    YARN = "yarn"
