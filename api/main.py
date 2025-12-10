from __future__ import annotations

"""
FastAPI app that loads TorchScript model and serves /predict.
"""

import logging
from pathlib import Path
from typing import List
import gdown

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .inference import Predictor

# Paths
TS_PATH = Path("models/best_model.ts")
CKPT_PATH = Path("src/checkpoints/best_model_gpu.pth")

TS_URL = "https://drive.google.com/uc?id=1p31v6AtVpfWuxXbSEKZmsAm6uIiiDjO6"
CKPT_URL = "https://drive.google.com/uc?id=1NrPv01afH327UcfsDcIgKJfrHHyCJ0Vo"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

def download_model():
    TS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TS_PATH.exists():
        logger.info(f"Downloading TorchScript model from {TS_URL}")
        gdown.download(TS_URL, str(TS_PATH), quiet=False)
    if not CKPT_PATH.exists():
        logger.info(f"Downloading checkpoint from {CKPT_URL}")
        gdown.download(CKPT_URL, str(CKPT_PATH), quiet=False)

download_model()
class PredictResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted class label.")
    confidence_scores: List[float] = Field(..., description="Softmax probabilities aligned with class_names.")
    class_names: List[str] = Field(..., description="List of class names used by the model.")
    gradcam_overlay: str = Field(..., description="Base64-encoded PNG of Grad-CAM overlay.")
    gradcam_layer: str = Field(..., description="Exact layer name used for Grad-CAM hooks.")
    gradcam_heatmap: str = Field(..., description="Base64-encoded PNG of raw Grad-CAM heatmap (grayscale or colored).")


def create_app() -> FastAPI:
    app = FastAPI(title="Brain Tumor Classification API", version="1.0.0")

    # Predictor is loaded at startup
    app.state.predictor = None

    @app.on_event("startup")
    def _load_models() -> None:
        try:
            app.state.predictor = Predictor(ts_path=TS_PATH, checkpoint_path=CKPT_PATH, num_classes=4)
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load models.")
            raise

    @app.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)) -> JSONResponse:
        """
        Run inference on an uploaded MRI image.
        """
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        try:
            data = await file.read()
            result = app.state.predictor.predict(data)
            return JSONResponse(content=result)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Prediction failed.")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()