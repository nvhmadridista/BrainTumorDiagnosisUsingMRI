# app/streamlit_app.py

import streamlit as st
import requests
from PIL import Image
import io
import base64

API_PREDICT = "http://localhost:8000/predict/"
API_GRADCAM = "http://localhost:8000/gradcam/"

st.title("Phân loại u não từ ảnh MRI")

uploaded_file = st.file_uploader("Tải lên ảnh MRI (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh đã upload
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

    # Nút dự đoán
    if st.button("Dự đoán"):
        with st.spinner("Đang xử lý dự đoán..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(API_PREDICT, files=files)
                response.raise_for_status()
                data = response.json()

                # Hiển thị kết quả
                st.success(f"Kết quả dự đoán: {data['predicted_class']}")

                st.write("Xác suất softmax:")
                for i, prob in enumerate(data['softmax_probabilities']):
                    st.write(f"Lớp {i}: {prob:.4f}")

            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi API: {e}")
    
    # Nút tạo Grad-CAM
    if st.button("Tạo Grad-CAM Heatmap"):
        with st.spinner("Đang tạo Grad-CAM..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

            try:
                response = requests.post(API_GRADCAM, files=files)
                response.raise_for_status()
                data = response.json()

                gradcam_b64 = data["gradcam_image"]

                gradcam_bytes = base64.b64decode(gradcam_b64)
                gradcam_img = Image.open(io.BytesIO(gradcam_bytes))

                st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)

            except Exception as e:
                st.error(f"Lỗi API: {e}")
