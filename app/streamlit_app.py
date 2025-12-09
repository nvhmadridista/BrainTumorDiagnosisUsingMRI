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

API_PREDICT_GRADCAM = "http://localhost:8000/predict_with_gradcam/"

st.set_page_config(layout="wide")
st.title("Phân loại u não từ ảnh MRI")

uploaded_file = st.file_uploader("Tải lên ảnh MRI (JPEG/PNG)", type=["jpg", "jpeg", "png"])

col_left, col_mid, col_right = st.columns([1, 1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with col_left:
        st.image(image, caption="Ảnh đã tải lên", width=280)

    predict_btn = st.button("Dự đoán")

    if predict_btn:
        with st.spinner("Đang xử lý..."):  
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
            try:
                response = requests.post(API_PREDICT_GRADCAM, files=files)
                response.raise_for_status()
                data = response.json()

                pred_class = data.get("class")
                softmax_probs = data.get("softmax", [])
                gradcam_b64 = data.get("gradcam")

                pred_class_name = CLASS_NAMES[pred_class] if isinstance(pred_class, int) and pred_class < len(CLASS_NAMES) else str(pred_class)

                gradcam_img = None
                if gradcam_b64:
                    gradcam_img = Image.open(io.BytesIO(base64.b64decode(gradcam_b64)))

            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi: {e}")
                if e.response is not None:
                    st.text(e.response.text)
                gradcam_img = None
                softmax_probs = []
                pred_class_name = ""
                
        # Ảnh
        with col_mid:
            st.subheader("Kết quả dự đoán")
            st.success(f"{pred_class_name}")

            st.write("Xác suất softmax:")
            for i, prob in enumerate(softmax_probs):
                class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Lớp {i}"
                st.write(f"{class_name}: {prob:.4f}")

        with col_right:
            st.subheader("Grad-CAM")
            if gradcam_img:
                st.image(gradcam_img, width=280)
            else:
                st.error("Không nhận được ảnh Grad-CAM")
        