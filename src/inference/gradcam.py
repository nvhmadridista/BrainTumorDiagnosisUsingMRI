import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        # nơi lưu feature map & gradient
        self.activations = None
        self.gradients = None

        # gắn hook
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            # grad_out chứa gradient của output layer
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        # Forward
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Activation + gradient 
        activations = self.activations.detach()      # (C, h, w)
        gradients = self.gradients.detach()          # (C, h, w)

        # Global Average Pool gradient 
        weights = gradients.mean(dim=(1, 2))         # (C)

        # Weighted sum 
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = torch.clamp(cam, min=0)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize về input size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )[0, 0].cpu().numpy()

        return cam

def overlay_heatmap(heatmap, img, alpha=0.5):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return blended
