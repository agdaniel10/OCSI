"""Re-ID appearance embedder (Phase 1).

Wraps a torchvision CNN backbone (default ResNet-18) as a person-appearance
feature extractor: crop each detection box, resize to the standard Re-ID aspect
ratio, ImageNet-normalise, run the backbone and L2-normalise the pooled feature.
The resulting ``(d,)`` vectors are what populate a track's appearance gallery and
long-term prototype ``ā`` — i.e. they are the signal the memory-similarity cue
(paper §4.2) needs in order to beat the geometry-only baseline.

torch/torchvision are imported lazily inside the constructor so that importing
:mod:`ocsi.perception` (and the model-free core) never requires them.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..config import PerceptionConfig, resolve_device
from ..types import Detection

# ImageNet normalisation (RGB)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class ReIDEmbedder:
    """CNN appearance embedder producing L2-normalised features.

    ``frame`` inputs are ``HxWx3`` uint8 **RGB** arrays (convert BGR → RGB before
    calling; :meth:`from_bgr` is a convenience). Output rows are unit-norm float32.
    """

    def __init__(self, cfg: Optional[PerceptionConfig] = None):
        import torch
        from torchvision import models

        self.cfg = cfg or PerceptionConfig()
        self._torch = torch
        self.device_name = resolve_device(self.cfg.device)
        self.device = torch.device(self.device_name)
        self.input_hw = tuple(self.cfg.reid_input_hw)

        weights = "DEFAULT" if self.cfg.reid_pretrained else None
        try:
            backbone = getattr(models, self.cfg.reid_backbone)(weights=weights)
        except AttributeError as e:
            raise ValueError(f"unknown torchvision backbone {self.cfg.reid_backbone!r}") from e
        except Exception as e:  # weight download failed (offline) -> random init
            if weights is None:
                raise
            warnings.warn(
                f"Re-ID pretrained weights unavailable ({e}); falling back to random init. "
                "Metrics will be meaningless until real weights are provided.",
                RuntimeWarning,
            )
            backbone = getattr(models, self.cfg.reid_backbone)(weights=None)

        # drop the classification head; keep the global-pooled feature
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.to(self.device).eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_hw[0], self.input_hw[1], device=self.device)
            self.dim = int(self.model(dummy).reshape(1, -1).shape[1])

    # ------------------------------------------------------------------ crops
    def _crop_chw(self, frame_rgb: np.ndarray, tlwh: np.ndarray) -> np.ndarray:
        """Clamp a box to the image, crop, resize and normalise -> (3, H, W) float32."""
        h_img, w_img = frame_rgb.shape[:2]
        x, y, w, h = (float(v) for v in tlwh)
        x1 = int(np.clip(round(x), 0, max(w_img - 1, 0)))
        y1 = int(np.clip(round(y), 0, max(h_img - 1, 0)))
        x2 = int(np.clip(round(x + w), x1 + 1, w_img))
        y2 = int(np.clip(round(y + h), y1 + 1, h_img))
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:                       # degenerate box: 1px fallback
            crop = frame_rgb[y1:y1 + 1, x1:x1 + 1]
        crop = cv2.resize(crop, (self.input_hw[1], self.input_hw[0]))  # cv2 takes (W, H)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - _MEAN) / _STD
        return np.transpose(crop, (2, 0, 1))

    # ------------------------------------------------------------- inference
    def __call__(self, frame_rgb: np.ndarray, boxes_tlwh: Sequence[np.ndarray]) -> np.ndarray:
        """Embed each box in ``frame_rgb`` -> ``(N, dim)`` L2-normalised float32."""
        boxes = np.asarray(boxes_tlwh, dtype=float).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        torch = self._torch
        crops = [self._crop_chw(frame_rgb, b) for b in boxes]
        bs = max(int(self.cfg.reid_batch_size), 1)
        feats: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(crops), bs):
                batch = torch.from_numpy(np.stack(crops[i:i + bs])).to(self.device)
                out = self.model(batch).reshape(batch.shape[0], -1)
                feats.append(out.cpu().numpy())
        feats_arr = np.concatenate(feats, axis=0).astype(np.float32)
        norms = np.linalg.norm(feats_arr, axis=1, keepdims=True)
        return feats_arr / np.clip(norms, 1e-12, None)

    def from_bgr(self, frame_bgr: np.ndarray, boxes_tlwh: Sequence[np.ndarray]) -> np.ndarray:
        """Convenience for OpenCV BGR frames."""
        return self(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), boxes_tlwh)

    def attach_embeddings(
        self, frame_rgb: np.ndarray, detections: Sequence[Detection]
    ) -> List[Detection]:
        """Compute and attach an embedding to each detection in place; returns them."""
        dets = list(detections)
        feats = self(frame_rgb, [d.tlwh for d in dets])
        for d, f in zip(dets, feats):
            d.embedding = f
        return dets
