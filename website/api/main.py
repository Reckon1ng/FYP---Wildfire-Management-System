from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import uvicorn
import base64
import io

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import binary_opening, binary_closing

# Local imports
from model.unet_inference import BurnScarUNetInference
from model.burn_scar_inference import BurnScarInference, SentinelHubAPI

# Helper for matplotlib images
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor='#0f172a') # Dark bg match
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

BASE_DIR = Path(__file__).parent.parent
WEIGHTS_PATH = BASE_DIR / "model" / "best_unet_burn_scar.pth"

app = FastAPI(title="Burn Scar Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
bbox_inference_engine = BurnScarInference(str(WEIGHTS_PATH))
sentinel_api = SentinelHubAPI()

# --- Request Models ---
class BBoxRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    date_from: str
    date_to: str

# --- Endpoints ---

@app.post("/predict_with_bbox")
async def predict_with_bbox(req: BBoxRequest):
    bbox = [req.min_lon, req.min_lat, req.max_lon, req.max_lat]

    # 1. Download Sentinel-2 imagery (12 bands)
    try:
        bands = sentinel_api.download_imagery(
            bbox=bbox, date_from=req.date_from, date_to=req.date_to, max_cloud_cover=30
        )
    except Exception as e:
        raise HTTPException(500, f"Sentinel Hub Error: {str(e)}")

    # 2. Run AI Inference
    # prediction (0 or 1), confidence (0.0 to 1.0)
    prediction, confidence = bbox_inference_engine.predict(bands)

    # 3. Post-processing
    h, w = prediction.shape
    total_pixels = prediction.size
    
    # Clean up prediction with morphological operations
    burn_mask = (prediction == 1) & (confidence > 0.85)
    burn_mask = binary_opening(burn_mask, structure=np.ones((2, 2)))
    burn_mask = binary_closing(burn_mask, structure=np.ones((2, 2)))
    
    burned_pixels = int(np.sum(burn_mask))

    # =======================================================
    # NEW FEATURE: BURN SEVERITY ANALYSIS (NBR)
    # =======================================================
    # NBR = (NIR - SWIR) / (NIR + SWIR)
    # Band indices from SentinelHubAPI: B02(0), B03(1), B04(2), B08(3-NIR), B11(4-SWIR1), B12(5)
    nir = bands[3]
    swir = bands[4]
    
    # Calculate NBR
    with np.errstate(divide='ignore', invalid='ignore'):
        nbr = (nir - swir) / (nir + swir + 1e-8)
    
    # Classify Severity only within the burned area
    # -1 to 0.1: High Severity (typically post-fire low values)
    # 0.1 to 0.27: Moderate
    # > 0.27: Low Severity / Regrowth
    # Note: These thresholds are approximations for demo purposes
    
    severity_map = np.zeros_like(nbr)
    severity_map[:] = np.nan # Background
    
    # We only care about severity inside the predicted burn mask
    mask_indices = np.where(burn_mask == 1)
    
    # Create empty counts
    sev_counts = {"high": 0, "moderate": 0, "low": 0}
    
    if burned_pixels > 0:
        burned_nbr = nbr[mask_indices]
        
        # 3 = High, 2 = Moderate, 1 = Low
        sev_class = np.zeros_like(burned_nbr)
        sev_class[burned_nbr < 0.1] = 3      # High
        sev_class[(burned_nbr >= 0.1) & (burned_nbr < 0.27)] = 2 # Moderate
        sev_class[burned_nbr >= 0.27] = 1    # Low
        
        # Map back to 2D array for visualization
        severity_map[mask_indices] = sev_class
        
        # Calculate stats
        sev_counts["high"] = float(np.sum(sev_class == 3) / burned_pixels)
        sev_counts["moderate"] = float(np.sum(sev_class == 2) / burned_pixels)
        sev_counts["low"] = float(np.sum(sev_class == 1) / burned_pixels)

    # =======================================================
    # VISUALIZATION GENERATION
    # =======================================================
    
    # 1. RGB Image
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    # Brighten it up slightly
    rgb_vis = np.clip(rgb * 2.5, 0, 1) 
    
    fig_rgb = plt.figure(figsize=(5, 5))
    plt.imshow(rgb_vis)
    plt.axis('off')
    rgb_b64 = fig_to_base64(fig_rgb)

    # 2. Overlay (Red on RGB)
    fig_overlay = plt.figure(figsize=(5, 5))
    plt.imshow(rgb_vis)
    overlay = np.zeros((h, w, 4))
    overlay[burn_mask == 1] = [1, 0, 0, 0.4] # Red, semi-transparent
    plt.imshow(overlay)
    plt.axis('off')
    overlay_b64 = fig_to_base64(fig_overlay)

    # 3. Severity Map
    fig_sev = plt.figure(figsize=(5, 5))
    # Custom colormap: Transparent(0), Yellow(Low), Orange(Mod), DarkRed(High)
    colors = ['#00000000', '#fde047', '#f97316', '#7f1d1d'] # Transparent, Yellow, Orange, Red
    cmap_sev = ListedColormap(colors)
    
    # Plot background (black/dark)
    plt.imshow(np.zeros_like(burn_mask), cmap='gray', vmin=0, vmax=1)
    
    # Plot severity
    # We need to handle NaNs or 0s. 
    # Let's make a display map where 0=Background, 1=Low, 2=Mod, 3=High
    display_map = np.zeros_like(nbr)
    display_map[mask_indices] = severity_map[mask_indices]
    
    plt.imshow(display_map, cmap=cmap_sev, vmin=0, vmax=3, interpolation='nearest')
    plt.axis('off')
    # Add a custom legend manually if needed, or just let the UI handle it
    severity_b64 = fig_to_base64(fig_sev)

    return {
        "burned_pixels": burned_pixels,
        "total_pixels": total_pixels,
        "severity_breakdown": sev_counts,
        "rgb_base64": rgb_b64,
        "overlay_base64": overlay_b64,
        "severity_base64": severity_b64
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)