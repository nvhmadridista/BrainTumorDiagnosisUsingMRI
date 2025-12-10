from __future__ import annotations

"""
FastAPI app that loads checkpoint model and serves /gradcam.
"""

import logging
from pathlib import Path
from typing import Optional
import gdown

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .inference import GradCAMPredictor

CKPT_PATH = Path("src/checkpoints/best_model_gpu.pth")
CKPT_URL = "https://drive.google.com/uc?id=1NrPv01afH327UcfsDcIgKJfrHHyCJ0Vo"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-gradcam")


def download_checkpoint():
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CKPT_PATH.exists():
        logger.info(f"Downloading checkpoint from {CKPT_URL}")
        gdown.download(CKPT_URL, str(CKPT_PATH), quiet=False)

download_checkpoint()

class GradcamResponse(BaseModel):
    gradcam_overlay: str = Field(..., description="Base64-encoded PNG of Grad-CAM overlay.")
    gradcam_layer: str = Field(..., description="Exact layer name used for Grad-CAM hooks.")
    gradcam_heatmap: str = Field(..., description="Base64-encoded PNG of raw Grad-CAM heatmap (grayscale or colored).")

def create_app() -> FastAPI:
    app = FastAPI(title="Brain Tumor Classification API - GradCAM", version="1.0.0")

    app.state.predictor = None

    @app.on_event("startup")
    def _load_model() -> None:
        try:
            app.state.predictor = GradCAMPredictor(checkpoint_path=CKPT_PATH, num_classes=4)
            logger.info("Grad-CAM model loaded successfully.")
        except Exception:
            logger.exception("Failed to load Grad-CAM model.")
            raise

    @app.post("/gradcam", response_model=GradcamResponse)
    async def gradcam(file: UploadFile = File(...), target_index: Optional[int] = None) -> JSONResponse:
        """
        Generate Grad-CAM for uploaded MRI image.
        """
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        try:
            data = await file.read()
            result = app.state.predictor.gradcam(data, target_index)
            return JSONResponse(content=result)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Grad-CAM generation failed.")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
