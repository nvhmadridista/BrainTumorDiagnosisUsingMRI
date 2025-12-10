from __future__ import annotations

"""
Streamlit UI for Brain Tumor Diagnosis Using MRI.
"""

from io import BytesIO
from typing import Any, Dict, List

import base64
import os
import requests
import streamlit as st
from PIL import Image

try:
    st.set_page_config(page_title="Brain Tumor Diagnosis Using MRI", layout="wide")
except Exception:
    pass

DEFAULT_PREDICT_URL = "https://brain-tumor-diagnosis-predict-api.onrender.com/predict"
DEFAULT_GRADCAM_URL = "https://nvhmadridista-Brain-Tumor-Diagnosis.hf.space/gradcam"

API_PREDICT_URL = os.environ.get("API_PREDICT_URL", DEFAULT_PREDICT_URL)
API_GRADCAM_URL = os.environ.get("API_GRADCAM_URL", DEFAULT_GRADCAM_URL)


def decode_base64_png(b64: str) -> Image.Image:
    data = base64.b64decode(b64.encode("utf-8"))
    return Image.open(BytesIO(data)).convert("RGB")


def main() -> None:
    st.title("Brain Tumor Diagnosis Using MRI")
        
    uploaded = st.file_uploader("Upload MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Please upload an MRI image.")
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input Image")
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

    if st.button("Predict"):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
        with st.spinner("Predicting..."):
            resp_pred = requests.post(API_PREDICT_URL, files=files, timeout=100)

        if resp_pred.status_code != 200:
            st.error(f"Predict request failed: {resp_pred.status_code} - {resp_pred.text}")
            return

        data_pred: Dict[str, Any] = resp_pred.json()
        pred_class: str = data_pred["predicted_class"]
        scores: List[float] = data_pred["confidence_scores"]
        class_names: List[str] = data_pred["class_names"]

        with st.spinner("Generating Grad-CAM..."):
            resp_cam = requests.post(API_GRADCAM_URL, files=files, timeout=60)

        if resp_cam.status_code != 200:
            st.warning(f"Grad-CAM request failed: {resp_cam.status_code} - {resp_cam.text}")
            overlay_b64 = None
        else:
            data_cam: Dict[str, Any] = resp_cam.json()
            overlay_b64 = data_cam.get("gradcam_overlay")

        with col2:
            st.subheader("Prediction")
            st.markdown(f"Predicted: **{pred_class}**")

            st.subheader("Confidence")
            for name, score in zip(class_names, scores):
                sc = float(score)
                pct = int(max(0.0, min(1.0, sc)) * 100)
                st.write(f"{name}: {sc:.3f}")
                st.progress(pct)

            st.subheader("Grad-CAM")
            if overlay_b64:
                overlay_img = decode_base64_png(overlay_b64)
                st.image(overlay_img, caption="Grad-CAM Overlay", use_column_width=True)
            else:
                st.info("Grad-CAM overlay unavailable.")

if __name__ == "__main__":
    main()