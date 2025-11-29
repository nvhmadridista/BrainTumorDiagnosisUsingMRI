import streamlit as st
import requests
from PIL import Image
import io
import base64
import json
from pathlib import Path

JSON_PATH = Path(__file__).resolve().parents[1] / "src" / "utils" / "class_names.json"
with open(JSON_PATH) as f:
    CLASS_NAMES = json.load(f)

API_PREDICT = "http://localhost:8000/predict/"
API_GRADCAM = "http://localhost:8000/gradcam/"

st.title("Phân loại u não từ ảnh MRI")

uploaded_file = st.file_uploader("Tải lên ảnh MRI (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", width=400)

    if st.button("Dự đoán"):
        with st.spinner("Đang xử lý dự đoán..."):  
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
            try:
                response = requests.post(API_PREDICT, files=files)
                response.raise_for_status()
                data = response.json()

                pred_class = data.get("class")
                softmax_probs = data.get("softmax")

                if pred_class is None or softmax_probs is None:
                    st.error("API trả về dữ liệu không hợp lệ.")
                else:
                    if isinstance(pred_class, int) and pred_class < len(CLASS_NAMES):
                        pred_class_name = CLASS_NAMES[pred_class]
                    else:
                        pred_class_name = pred_class

                    st.success(f"Kết quả dự đoán: {pred_class_name}")
                    st.write("Xác suất softmax:")

                    for i, prob in enumerate(softmax_probs):
                        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Lớp {i}"
                        st.write(f"{class_name}: {prob:.4f}")

            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi API: {e}")
                if e.response is not None:
                    st.text(e.response.text)
    
    if st.button("Tạo Grad-CAM Heatmap"):
        with st.spinner("Đang tạo Grad-CAM..."):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}

            try:
                response = requests.post(API_GRADCAM, files=files)
                response.raise_for_status()
                data = response.json()

                gradcam_b64 = data.get("gradcam")
                if gradcam_b64 is None:
                    st.error("API trả về dữ liệu Grad-CAM không hợp lệ.")
                else:
                    gradcam_bytes = base64.b64decode(gradcam_b64)
                    gradcam_img = Image.open(io.BytesIO(gradcam_bytes))
                    st.image(gradcam_img, caption="Grad-CAM Heatmap", width=400)

            except Exception as e:
                st.error(f"Lỗi API: {e}")
                if e.response is not None:
                    st.text(e.response.text)
