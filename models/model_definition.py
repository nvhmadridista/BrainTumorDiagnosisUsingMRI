from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid_model import BrainTumorModel  


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pytorch_model(
    checkpoint_path: Path,
    num_classes: int = 4,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = get_device()

    model = BrainTumorModel(num_classes=num_classes)
    model.to(device)
    model.eval()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model