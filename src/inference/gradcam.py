import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        # output: (1, C, H, W)
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        # grad_output[0]: (1, C, H, W)
        self.gradients = grad_output[0].detach()

    def generate(self, x):
        # Forward 
        logits = self.model(x)
        class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]

        # Backward
        self.model.zero_grad()
        score.backward()

        # Lấy gradient & activation
        gradients = self.gradients      # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # (1, H, W)
        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze()

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

    return overlay
