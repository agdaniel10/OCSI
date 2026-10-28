"""Detection attribution producers (X-OCSI Eq. 2-3).

Turns a detector's Grad-CAM into a full-frame attribution heatmap per detection.
The map is defined over the *whole source frame*, not cropped to the box, so
downstream spatial checks (soft IoU against the activity map, or energy-in-box
when only one attribution map exists) measure real localisation rather than a
tautology.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .gradcam import DetectorGradCAM
from .utils import normalize_heatmap, resize_heatmap


def feature_activation_cam(feature_map: np.ndarray) -> np.ndarray:
    """Simple CAM from feature map activations (fallback when gradients unavailable)."""
    fm = np.asarray(feature_map, dtype=float)
    if fm.ndim == 3:
        cam = np.mean(fm, axis=0)
    elif fm.ndim == 2:
        cam = fm
    else:
        return np.zeros((1, 1), dtype=float)
    return normalize_heatmap(cam)


def spatial_attribution_from_box(
    heatmap: np.ndarray, bbox_tlwh: np.ndarray, img_shape: Tuple[int, int]
) -> np.ndarray:
    """Return a normalised full-frame attribution map at image resolution."""
    del bbox_tlwh  # kept for API compatibility: the map covers the whole frame
    H, W = img_shape[:2]
    hm = normalize_heatmap(heatmap)
    if hm.shape != (H, W):
        hm = resize_heatmap(hm, (H, W))
    return hm


def letterbox_image(image_bgr: np.ndarray, size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize+pad a BGR image to a square ``size`` tensor input.

    Returns ``(float_rgb_chw, scale, (pad_x, pad_y))`` where ``scale`` maps source
    pixels to resized pixels and the padding is split evenly on both sides.
    """
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("letterbox_image expects an (H, W, 3) BGR image")
    h, w = img.shape[:2]
    scale = float(size) / float(max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    try:
        import cv2

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    except ImportError:  # pragma: no cover - cv2 is a perception extra
        rows = np.clip((np.arange(nh) * h / nh).astype(int), 0, h - 1)
        cols = np.clip((np.arange(nw) * w / nw).astype(int), 0, w - 1)
        resized = img[rows[:, None], cols[None, :]]

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return chw, scale, (pad_x, pad_y)


def boxes_to_input(boxes_tlwh: Sequence[np.ndarray], scale: float, pad: Tuple[int, int]) -> np.ndarray:
    """Map ``(tlwh)`` boxes from source pixels to letterboxed input pixels."""
    boxes = np.asarray(boxes_tlwh, dtype=float).reshape(-1, 4)
    px, py = pad
    out = np.empty((len(boxes), 4), dtype=float)
    out[:, 0] = boxes[:, 0] * scale + px
    out[:, 1] = boxes[:, 1] * scale + py
    out[:, 2] = (boxes[:, 0] + boxes[:, 2]) * scale + px
    out[:, 3] = (boxes[:, 1] + boxes[:, 3]) * scale + py
    return out


def unletterbox_heatmaps(
    cams: Sequence[np.ndarray],
    scale: float,
    pad: Tuple[int, int],
    image_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """Crop the letterbox padding and resize each CAM back to source resolution."""
    H, W = int(image_shape[0]), int(image_shape[1])
    px, py = pad
    nh, nw = int(round(H * scale)), int(round(W * scale))
    out: List[np.ndarray] = []
    for cam in cams:
        cam = np.asarray(cam, dtype=float)
        content = cam[py:py + nh, px:px + nw]
        if content.size == 0:
            content = cam
        out.append(normalize_heatmap(resize_heatmap(content, (H, W))))
    return out


def detection_heatmaps(
    model,
    image_bgr: np.ndarray,
    boxes_tlwh: Sequence[np.ndarray],
    class_ids: Optional[Sequence[int]] = None,
    size: int = 640,
    head=None,
    num_classes: Optional[int] = None,
    heatmap_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Full-frame detector attribution ``E_det`` for each box in ``boxes_tlwh``.

    ``heatmap_size`` downsamples the returned maps (default: source resolution).
    Attribution is cached per detection, so full-resolution maps cost tens of MB
    per frame on a 1080p sequence; ``(224, 224)`` keeps the spatial structure while
    making the cache practical. Boxes and ``img_shape`` remain in source pixels.
    """
    if len(boxes_tlwh) == 0:
        return []
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a perception extra
        raise ImportError("detection attribution needs PyTorch") from exc

    chw, scale, pad = letterbox_image(image_bgr, size)
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    dtype = params[0].dtype if params else torch.float32
    if device.type == "cpu":
        # Grad-CAM needs a backward pass; bfloat16/float16 on CPU is unsupported.
        dtype = torch.float32
    tensor = torch.from_numpy(chw)[None].to(device=device, dtype=dtype)
    cam = DetectorGradCAM(model, head=head, num_classes=num_classes)
    input_boxes = boxes_to_input(boxes_tlwh, scale, pad)
    raw = cam.cam_for_boxes(tensor, input_boxes, class_ids=class_ids)
    H, W = np.asarray(image_bgr).shape[:2]
    maps = unletterbox_heatmaps(raw, scale, pad, (H, W))
    if heatmap_size is not None:
        maps = [resize_heatmap(m, (int(heatmap_size[0]), int(heatmap_size[1]))) for m in maps]
    return maps


def explanation_map_summary(heatmap: np.ndarray) -> dict:
    """Compact descriptor stored in the feature cache instead of the full map."""
    hm = np.asarray(heatmap, dtype=float)
    H, W = hm.shape
    ys, xs = np.nonzero(hm > (0.5 * hm.max() if hm.max() > 0 else np.inf))
    if len(ys) == 0:
        ys, xs = np.unravel_index([int(np.argmax(hm))], hm.shape)
    return {
        "shape": (int(H), int(W)),
        "centroid": (float(xs.mean()) / max(W - 1, 1), float(ys.mean()) / max(H - 1, 1)),
        "peak": float(hm.max()),
        "energy": float(hm.sum()),
    }
