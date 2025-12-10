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

DEFAULT_API_URL = "https://brain-tumor-api.onrender.com/predict"
API_URL = os.environ.get("API_URL", DEFAULT_API_URL)


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
            resp = requests.post(API_URL, files=files, timeout=60)

        if resp.status_code != 200:
            st.error(f"Request failed: {resp.status_code} - {resp.text}")
            return

        data: Dict[str, Any] = resp.json()
        pred_class: str = data["predicted_class"]
        scores: List[float] = data["confidence_scores"]
        class_names: List[str] = data["class_names"]
        overlay_b64: str = data["gradcam_overlay"]

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
            overlay_img = decode_base64_png(overlay_b64)
            st.image(overlay_img, caption="Grad-CAM Overlay", use_column_width=True)

if __name__ == "__main__":
    main()