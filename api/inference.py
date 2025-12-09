from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image

from .utils import build_preprocess, CLASS_NAMES, get_device, pil_to_base64_png, load_image_from_bytes, DEFAULT_SIZE
from .gradcam import GradCAM, overlay_heatmap
from models.model_definition import load_pytorch_model

class Predictor:
    def __init__(self, ts_path: Path, checkpoint_path: Path, num_classes: int = 4) -> None:
        self.device = get_device()

        if not ts_path.exists():
            raise FileNotFoundError(f"TorchScript model not found: {ts_path}")
        self.ts_model = torch.jit.load(str(ts_path), map_location=self.device)
        self.ts_model.eval()

        self.grad_model = load_pytorch_model(checkpoint_path=checkpoint_path, num_classes=num_classes, device=self.device)
        self.grad_model.eval()

        self.preprocess = build_preprocess()

        target_layer = self.grad_model.stage3
        self.cam = GradCAM(self.grad_model, target_layer)

    def preprocess_image(self, pil_img: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image to tensor (1, 3, 224, 224).
        """
        x = self.preprocess(pil_img)
        return x.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer_logits(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.ts_model(input_tensor)
        return logits

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Complete prediction pipeline:
        - Preprocess
        - Inference 
        - Softmax scores, predicted class
        - Grad-CAM overlay 

        Returns:
            Dict with predicted_class, confidence_scores, class_names, gradcam_overlay
        """
        pil_img = load_image_from_bytes(image_bytes)
        input_tensor = self.preprocess_image(pil_img)
        resized_pil = pil_img.resize(DEFAULT_SIZE, resample=Image.BILINEAR)

        logits = self.grad_model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence_scores: List[float] = probs.squeeze(0).tolist()
        predicted_class = CLASS_NAMES[pred_idx]

        heatmap = self.cam.generate_cam(input_tensor, pred_idx)
        overlay = overlay_heatmap(heatmap, resized_pil)

        return {
            "predicted_class": predicted_class,
            "confidence_scores": confidence_scores,
            "class_names": CLASS_NAMES,
            "gradcam_overlay": pil_to_base64_png(overlay),
        }