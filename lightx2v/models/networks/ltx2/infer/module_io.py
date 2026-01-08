from dataclasses import dataclass
from typing import Tuple, Optional

import torch


@dataclass
class GridOutput:
    """Output container for grid size information."""
    tensor: torch.Tensor
    tuple: Tuple[int, int, int]


@dataclass
class LTX2PreInferModuleOutput:
    """Output container for LTX2 pre-inference module."""
    x: torch.Tensor  # Hidden states
    embed: torch.Tensor  # Time embedding
    context: torch.Tensor  # Text conditioning
    grid_sizes: GridOutput  # Grid dimensions
    cos_sin: Optional[torch.Tensor] = None  # Rotary position embeddings
    adapter_args: Optional[dict] = None  # Additional adapter arguments
