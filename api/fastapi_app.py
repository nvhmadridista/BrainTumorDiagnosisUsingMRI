from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import base64
import cv2
import torch

from src.inference.preprocess import preprocess
from src.inference.load_onnx_model import ONNXModel
from src.inference.load_pytorch_model import load_torch_model, predict_torch
from src.inference.gradcam import GradCAM, overlay_heatmap

app = FastAPI(title="API Phân loại U não từ MRI")

# Khởi tạo model ONNX và PyTorch
onnx_path = "deploy/model.onnx"
pth_path = "deploy/best_model.pth"

onnx_model = ONNXModel(onnx_path)
torch_model = load_torch_model(pth_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model.to(device)
torch_model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...), use_onnx: bool = True):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File phải là ảnh JPEG hoặc PNG")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể mở ảnh")

    input_tensor = preprocess(image)

    if use_onnx:
        pred_class, softmax_probs = onnx_model.predict(input_tensor)
    else:
        pred_class, softmax_probs = predict_torch(torch_model, input_tensor, device=device)

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
    input_tensor_torch = torch.from_numpy(input_tensor).to(device)

    target_layer = torch_model.stage4[-1]   # layer cuối của backbone
    
    cam = GradCAM(torch_model, target_layer)
    heatmap = cam.generate(input_tensor)
    overlay = overlay_heatmap(heatmap, image)

    _, buffer = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buffer).decode()

    return {"gradcam": b64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
