import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        target_layer.register_forward_hook(self.forward_hook)
        # Use full backward hook to avoid deprecation and capture correct grads
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        # output: (1, C, H, W)
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        # grad_output[0]: (1, C, H, W)
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_index):
        """
        Generate Grad-CAM heatmap for a given input and target class index.

        Args:
            input_tensor: torch.Tensor of shape (1, C, H, W)
            target_index: int, target class index to compute Grad-CAM for
        """
        # Forward pass
        output = self.model(input_tensor)  # (1, num_classes)

        # Backward pass w.r.t. the target class score
        self.model.zero_grad()
        target_score = output[0, target_index]
        target_score.backward(retain_graph=True)

        gradients = self.gradients      # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # (1, H, W)
        cam = torch.relu((weights * activations).sum(dim=1)).squeeze()

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # 224x224
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        return cam.cpu().numpy()

def overlay_heatmap(cam, pil_image):
    image = np.array(pil_image.resize((224, 224)))
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    return Image.fromarray(overlay)