from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import cv2
import numpy as np

from src.inference.preprocess import preprocess
from src.inference.load_model import ONNXModel
from src.inference.gradcam import make_gradcam_heatmap, overlay_heatmap
#from src.models.hybrid_model import build_hybrid_model

app = FastAPI(title="API Phân loại U não từ MRI")

# Khởi tạo model ONNX 
MODEL_PATH = "deploy/model.onnx"
model = ONNXModel(MODEL_PATH)

# Khởi tạo model Keras (dùng cho Grad-CAM)
#keras_model = build_hybrid_model()
keras_model = None

# Cần có model thật đặt ở deploy/
keras_model.load_weights("deploy/keras_weights.h5")  # Đường dẫn file weights Keras
last_conv_layer_name = "convnext_tiny/block_4/project_conv"  # Tên layer conv cuối cùng của backbone?

def pil_image_to_base64(img: np.ndarray) -> str:
    """
    Chuyển ảnh numpy RGB thành base64 PNG string
    """
    _, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Kiểm tra file ảnh
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File phải là ảnh JPEG hoặc PNG")

    # Đọc ảnh từ upload
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể mở ảnh")

    # Tiền xử lý
    input_tensor = preprocess(image)

    # Dự đoán
    try:
        pred_class, softmax_probs = model.predict(input_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {e}")

    # Tạo Grad-CAM heatmap với model Keras
    try:
        heatmap = make_gradcam_heatmap(input_tensor, keras_model, last_conv_layer_name, pred_class)
        img_np = np.array(image.resize((224, 224)))
        overlay_img = overlay_heatmap(img_np, heatmap)
        heatmap_base64 = pil_image_to_base64(overlay_img)
    except Exception as e:
        heatmap_base64 = ""

    return JSONResponse(content={
        "predicted_class": int(pred_class),
        "softmax_probabilities": softmax_probs.tolist(),
         "gradcam_heatmap": heatmap_base64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
