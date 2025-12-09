from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .model_definition import load_pytorch_model, example_input, get_device


DEFAULT_CHECKPOINT = Path("src/checkpoints/best_model_gpu.pth")
DEFAULT_TS_PATH = Path("models/best_model.ts")


def export_to_torchscript(
    model: nn.Module,
    save_path: Path,
    use_script: bool = True,
    dummy_input: Optional[torch.Tensor] = None,
) -> Path:
    """
    Export a model to TorchScript via script (preferred) or trace.

    Args:
        model: Loaded PyTorch model (eval mode).
        save_path: Output .ts path.
        use_script: If True, use torch.jit.script; otherwise torch.jit.trace.
        dummy_input: Optional dummy input for tracing.

    Returns:
        Path to the saved TorchScript file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if use_script:
        try:
            ts = torch.jit.script(model)
        except Exception:
            use_script = False
        else:
            ts.save(str(save_path))
            return save_path

    if not use_script:
        if dummy_input is None:
            dummy_input = example_input(1).to(next(model.parameters()).device)
        ts = torch.jit.trace(model, dummy_input, strict=False, check_trace=False)
        ts.save(str(save_path))
        return save_path

    raise RuntimeError("TorchScript export failed.")


if __name__ == "__main__":
    device = get_device()
    model = load_pytorch_model(checkpoint_path=DEFAULT_CHECKPOINT, num_classes=4, device=device)
    model.eval()
    out = export_to_torchscript(model, DEFAULT_TS_PATH, use_script=True)
    print(f"TorchScript saved to: {out.resolve()}")