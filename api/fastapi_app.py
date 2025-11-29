from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import base64
import cv2
import torch
import numpy as np

from src.inference.preprocess import preprocess, resize_for_overlay
from src.inference.load_pytorch_model import load_torch_model, predict_torch
from src.inference.gradcam import GradCAM, overlay_heatmap

app = FastAPI(title="API Phân loại U não từ MRI")

# Khởi tạo model PyTorch
pth_path = "deploy/best_model.pth"
torch_model = load_torch_model(pth_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model.to(device)
torch_model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File phải là ảnh JPEG hoặc PNG")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể mở ảnh")

    input_tensor = preprocess(image)

    if isinstance(input_tensor, np.ndarray):
        input_tensor_torch = torch.from_numpy(input_tensor).to(device)
    else:
        input_tensor_torch = input_tensor.to(device)

    pred_class, softmax_probs = predict_torch(torch_model, input_tensor_torch)

    return {
        "class": int(pred_class),
        "softmax": softmax_probs.tolist()
    }

@app.post("/gradcam/")
async def gradcam(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể mở ảnh")

    input_tensor = preprocess(image)
    
    if isinstance(input_tensor, np.ndarray):
        input_tensor_torch = torch.from_numpy(input_tensor).to(device)
    else:
        input_tensor_torch = input_tensor.to(device)

    target_layer = torch_model.stage3   # chọn stage, block hay layer trong backbone
    
    cam = GradCAM(torch_model, target_layer)
    heatmap = cam.generate(input_tensor_torch)

    image_for_overlay = resize_for_overlay(image)
    overlay = overlay_heatmap(heatmap, image_for_overlay)

    _, buffer = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buffer).decode()

    return {"gradcam": b64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
