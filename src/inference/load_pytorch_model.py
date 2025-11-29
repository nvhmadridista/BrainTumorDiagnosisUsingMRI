import torch
import numpy as np
from src.models.hybrid_model import BrainTumorModel 

def load_torch_model(model_path: str, device='cpu'):
    model = BrainTumorModel(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_torch(model, input_tensor: np.ndarray, device='cpu'):
    input_tensor = input_tensor.to(device)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)  # shape (1, num_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_class = int(probs.argmax(axis=1)[0])
    return pred_class, probs[0]
