"""
Burn Scar Detection Inference Script
"""
import torch
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import segmentation_models_pytorch as smp
import requests
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

class BurnScarInference:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=8,
            classes=2,
            activation=None
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded from {model_path}")
    
    def preprocess_bands(self, bands: np.ndarray) -> torch.Tensor:
        blue, green, red, nir, swir1, swir2 = bands
        ndvi = (nir - red) / (nir + red + 1e-8)
        nbr = (nir - swir1) / (nir + swir1 + 1e-8)
        img_8ch = np.stack([blue, green, red, nir, swir1, swir2, ndvi, nbr], axis=0)
        img_tensor = torch.from_numpy(img_8ch).float()
        img_tensor = torch.clamp(img_tensor / 3000.0, 0, 1)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    
    def predict(self, bands: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_tensor = self.preprocess_bands(bands).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        # NO INVERSION - model already learned correct labels!
        # prediction = 1 - prediction  ← REMOVE THIS LINE
        confidence = probs[0, 1].cpu().numpy()  # Use class 1 probability
        
        # Debug
        print(f"Prediction - unique: {np.unique(prediction)}, sum: {np.sum(prediction)}")
        
        return prediction, confidence


    
    def visualize_results(self, bands: np.ndarray, prediction: np.ndarray, 
                         confidence: np.ndarray, save_path: Optional[str] = None):
        rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
        rgb_vis = np.zeros_like(rgb)
        for i in range(3):
            p2, p98 = np.percentile(rgb[:,:,i], (2, 98))
            rgb_vis[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2 + 1e-8), 0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_vis)
        axes[0].set_title('Satellite RGB')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title('Burn Scar Detection')
        axes[1].axis('off')
        
        im = axes[2].imshow(confidence, cmap='YlOrRd', vmin=0, vmax=1)
        axes[2].set_title('Confidence Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        plt.show()
    
    def calculate_metrics(self, prediction: np.ndarray, ground_truth: np.ndarray) -> Dict:
        from sklearn.metrics import accuracy_score, jaccard_score, f1_score, precision_score, recall_score
        
        print(f"\n=== INSIDE calculate_metrics ===")
        print(f"INPUT prediction - unique: {np.unique(prediction)}, sum: {np.sum(prediction)}, dtype: {prediction.dtype}")
        print(f"INPUT ground_truth - unique: {np.unique(ground_truth)}, sum: {np.sum(ground_truth)}, dtype: {ground_truth.dtype}")
        
        # Flatten
        pred_flat = prediction.flatten()
        gt_flat = ground_truth.flatten()
        print(f"After flatten - pred unique: {np.unique(pred_flat)}, gt unique: {np.unique(gt_flat)}")
        
        # Filter invalid
        valid_mask = (gt_flat != 255) & (gt_flat != -1)
        print(f"Valid mask - keeping {np.sum(valid_mask)} of {len(valid_mask)} pixels")
        
        pred_flat = pred_flat[valid_mask]
        gt_flat = gt_flat[valid_mask]
        print(f"After filtering - pred unique: {np.unique(pred_flat)}, sum: {np.sum(pred_flat)}")
        print(f"After filtering - gt unique: {np.unique(gt_flat)}, sum: {np.sum(gt_flat)}")
        
        metrics = {
            'accuracy': accuracy_score(gt_flat, pred_flat),
            'iou': jaccard_score(gt_flat, pred_flat, average='binary', zero_division=0),
            'f1': f1_score(gt_flat, pred_flat, average='binary', zero_division=0),
            'precision': precision_score(gt_flat, pred_flat, average='binary', zero_division=0),
            'recall': recall_score(gt_flat, pred_flat, average='binary', zero_division=0)
        }
        
        print(f"=== METRICS DONE ===\n")
        return metrics


class SentinelHubAPI:
    def __init__(self, client_id: str = None, client_secret: str = None):
        if client_id is None or client_secret is None:
            from config import SENTINEL_CLIENT_ID, SENTINEL_CLIENT_SECRET
            client_id = client_id or SENTINEL_CLIENT_ID
            client_secret = client_secret or SENTINEL_CLIENT_SECRET
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_url = 'https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token'
        self.process_url = 'https://services.sentinel-hub.com/api/v1/process'
        self._authenticate()
    
    def _authenticate(self):
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        response = requests.post(self.token_url, data=payload)
        response.raise_for_status()
        self.access_token = response.json()['access_token']
        print("✓ Authenticated with Sentinel Hub")
    
    def download_imagery(self, bbox: list, date_from: str, date_to: str, 
                        max_cloud_cover: float = 20.0) -> np.ndarray:
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "input": {
                "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [{"type": "sentinel-2-l2a", "dataFilter": {
                    "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                    "maxCloudCoverage": max_cloud_cover
                }}]
            },
            "output": {"width": 512, "height": 512, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": """
                //VERSION=3
                function setup() {
                    return {input: ["B02", "B03", "B04", "B08", "B11", "B12"], output: {bands: 6, sampleType: "FLOAT32"}};
                }
                function evaluatePixel(sample) {
                    return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
                }
            """
        }
        
        response = requests.post(self.process_url, headers=headers, json=payload)
        response.raise_for_status()
        
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                bands = dataset.read()
        
        print(f"✓ Downloaded imagery: {bands.shape}")
        return bands
