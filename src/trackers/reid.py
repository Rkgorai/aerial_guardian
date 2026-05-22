import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class HSVColorHistogramExtractor:
    """Robust, fast fallback appearance feature extractor using spatial HSV histograms.
    
    Splits the cropped pedestrian image vertically into 3 zones (head/shoulders, torso, legs)
    to capture spatial clothing color layout. Computes HSV histograms for each zone and
    concatenates them into a unified, L2-normalized embedding vector.
    """

    def __init__(self, h_bins=8, s_bins=8, v_bins=8):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins

    def extract(self, crop):
        """Extract spatial HSV histogram embedding from a cropped BGR image patch.
        
        crop: BGR image patch (numpy array)
        Returns: L2-normalized feature vector (numpy array) of size (3 * h_bins * s_bins * v_bins)
        """
        if crop is None or crop.size == 0:
            total_size = 3 * self.h_bins * self.s_bins * self.v_bins
            return np.zeros(total_size, dtype=np.float32)

        # Convert crop to HSV color space
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape

        # Vertical splits: 3 regions (top 20%, middle 40%, bottom 40%)
        y_bounds = [0, int(h * 0.2), int(h * 0.6), h]
        features = []

        for i in range(3):
            y_start = y_bounds[i]
            y_end = y_bounds[i + 1]
            if y_end <= y_start:
                hist_size = self.h_bins * self.s_bins * self.v_bins
                features.append(np.zeros(hist_size, dtype=np.float32))
                continue

            region = hsv[y_start:y_end, :]
            # Compute 3D HSV histogram
            hist = cv2.calcHist(
                [region],
                [0, 1, 2],
                None,
                [self.h_bins, self.s_bins, self.v_bins],
                [0, 180, 0, 256, 0, 256]
            )
            # Flatten and add to features
            features.append(hist.flatten())

        # Concatenate spatial histograms
        feature_vector = np.concatenate(features)
        
        # L2 Normalize
        norm = np.linalg.norm(feature_vector)
        if norm > 1e-5:
            feature_vector = feature_vector / norm
        else:
            feature_vector = np.zeros_like(feature_vector)

        return feature_vector


class PyTorchReIDExtractor:
    """Appearance feature extractor using a PyTorch CNN."""

    def __init__(self, model_name="resnet18", device="cpu"):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = None
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading PyTorch ReID Extractor: {self.model_name}...")
            if self.model_name == "resnet18":
                # We load resnet18, extract features from before the fully-connected layer
                weights = models.ResNet18_Weights.DEFAULT
                backbone = models.resnet18(weights=weights)
                self.model = nn.Sequential(*list(backbone.children())[:-1])  # remove avgpool & fc, output size: [B, 512, 1, 1]
            elif self.model_name == "mobilenet":
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                backbone = models.mobilenet_v3_small(weights=weights)
                self.model = backbone.features  # output shape: [B, 576, 4, 2]
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            self.model.to(self.device)
            self.model.eval()
            print("PyTorch ReID Extractor successfully loaded.")
        except Exception as e:
            print(f"Failed to load PyTorch ReID weights: {e}")
            print("Falling back to HSV Color Histogram Extractor.")
            self.model = None

    def extract(self, crop):
        """Extract normalized embedding vector from a cropped BGR image patch.
        
        crop: BGR image patch (numpy array)
        Returns: L2-normalized feature vector (numpy array)
        """
        if self.model is None or crop is None or crop.size == 0:
            return None

        try:
            # Crop preprocess
            # Convert BGR (OpenCV) to RGB for PyTorch
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(tensor)
                if self.model_name == "mobilenet":
                    features = self.pool(features)
                features = torch.flatten(features, 1)  # shape [1, D]
                # L2 Normalize
                features = nn.functional.normalize(features, p=2, dim=1)
                embedding = features.cpu().numpy()[0]

            return embedding
        except Exception as e:
            # Under any runtime error, return None to trigger fallback
            return None


class ReIDManager:
    """Unified appearance feature extractor. Handles PyTorch inference and HSV histogram fallbacks."""

    def __init__(self, model_name="resnet18", device="cpu"):
        self.device = device
        self.pytorch_extractor = PyTorchReIDExtractor(model_name=model_name, device=device)
        self.hsv_extractor = HSVColorHistogramExtractor()

    def extract(self, frame, bbox):
        """Extract ReID appearance embedding from a bounding box within a frame.
        
        frame: BGR frame (numpy array)
        bbox: bounding box in [x, y, w, h] format (center format or [x1, y1, x2, y2])
        Returns: normalized feature vector (numpy array)
        """
        if frame is None or len(bbox) < 4:
            return np.zeros(1, dtype=np.float32)

        # Convert bbox to [x1, y1, x2, y2]
        x, y, w, h = bbox[:4]
        # Calculate bounds
        x1 = max(0, int(x - w / 2))
        y1 = max(0, int(y - h / 2))
        x2 = min(frame.shape[1], int(x + w / 2))
        y2 = min(frame.shape[0], int(y + h / 2))

        if x2 <= x1 or y2 <= y1:
            return np.zeros(1, dtype=np.float32)

        # Crop patch
        crop = frame[y1:y2, x1:x2]

        # Attempt PyTorch extraction
        feat = self.pytorch_extractor.extract(crop)
        if feat is not None:
            return feat

        # Fallback to HSV color histogram
        return self.hsv_extractor.extract(crop)
