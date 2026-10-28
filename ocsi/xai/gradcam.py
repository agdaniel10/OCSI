"""Grad-CAM for classifiers and anchor-based detection heads.

Two entry points:

* :class:`GradCAM` — the original classifier CAM (logits -> argmax -> backward).
* :class:`DetectorGradCAM` — CAM for YOLO-style heads, where the model emits a
  ``(B, 4 + nc, A)`` (or ``5 + nc`` with objectness) tensor of per-anchor
  predictions rather than class logits. Attribution is requested *per box*, so
  ``E_det^y`` in the X-OCSI paper is evidence for one detection, not the whole
  frame.

Both are model-agnostic apart from the shape conventions documented on the
methods, and both degrade to ``None``/zeros rather than raising when gradients
are unavailable.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


class GradCAM:
    def __init__(self, model, target_layer=None):
        if torch is None:
            raise ImportError("PyTorch required for GradCAM")
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __enter__(self):
        if self.target_layer is None:
            raise ValueError("target_layer must be specified")
        self._hooks.append(self.target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        if isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return cam


def _as_pred_tensor(output):
    """Extract the ``(B, C, A)`` prediction tensor from a detector forward output.

    Handles the shapes seen in practice: a bare tensor, a tuple/list whose first
    element is the decoded prediction, or a dict with a ``"pred"``-style key.
    """
    if torch is not None and torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("pred", "preds", "predictions", "logits"):
            if key in output:
                return _as_pred_tensor(output[key])
        raise TypeError("detector output dict has no prediction tensor")
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch is not None and torch.is_tensor(item) and item.ndim == 3:
                return item
        if output:
            return _as_pred_tensor(output[0])
    raise TypeError(f"cannot interpret detector output of type {type(output)!r}")


class DetectorGradCAM:
    """Grad-CAM for anchor-based detection heads.

    The head is assumed to map a list of feature maps (one per stride) to a
    ``(B, C, A)`` prediction tensor, with ``A`` the concatenation of the per-level
    anchor grids in row-major order — the layout used by Ultralytics YOLO heads.
    Only the class/objectness channels are used, so no NMS or box decoding is
    needed to attribute a detection.

    Parameters
    ----------
    model:
        Callable module. Its forward output must be interpretable by
        :func:`_as_pred_tensor`.
    head:
        Module whose *input* activations are the CAM feature maps. Defaults to the
        detection head found by :meth:`_default_head` (a named ``head``/``detect``
        module, else the last layer of an inner ``model`` sequence).
    num_classes:
        Number of classes in the head. Inferred from ``head.nc`` when omitted.
    class_channel_offset:
        ``4`` for YOLOv8-style heads (no objectness) and ``5`` for YOLOv5-style
        heads (objectness at channel 4). Inferred from ``C`` when omitted.
    receptive_radius:
        Extra grid cells around a box's anchor footprint included in its mask,
        accounting for the 3x3 convs inside YOLO heads.
    """

    def __init__(
        self,
        model,
        head=None,
        num_classes: Optional[int] = None,
        class_channel_offset: Optional[int] = None,
        receptive_radius: int = 1,
    ):
        if torch is None:  # pragma: no cover - torch is an optional extra
            raise ImportError("PyTorch required for DetectorGradCAM")
        self.model = model
        self.head = head if head is not None else self._default_head(model)
        self.num_classes = num_classes
        self.class_channel_offset = class_channel_offset
        self.receptive_radius = int(receptive_radius)

    @staticmethod
    def _default_head(model):
        """Best-effort detection-head discovery.

        Order: an explicitly named head attribute (``head``/``detect``), then the
        last layer of an inner ``model`` sequence (Ultralytics), then the last
        direct child. Deliberately avoids ``modules()[-1]``, which descends into
        the head's own convolutions.
        """
        for name in ("head", "detect", "det_head"):
            candidate = getattr(model, name, None)
            if candidate is not None and not torch.is_tensor(candidate) and hasattr(
                candidate, "forward"
            ):
                return candidate
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "__getitem__"):
            try:
                return inner[-1]
            except (IndexError, TypeError, KeyError):
                pass
        children = list(model.children())
        return children[-1] if children else list(model.modules())[-1]

    # ----------------------------------------------------------- internals
    def _feature_maps(self, input_tensor):
        """Forward once, returning ``(preds, feature_maps)`` with grads retained.

        Ultralytics checkpoints are loaded with ``requires_grad=False`` on every
        parameter (inference default), which would make the whole graph inert.
        Gradients are enabled for the forward pass only and the previous flags are
        restored before this method returns.
        """
        captured: List = []

        def hook(_module, inputs, _output):
            feats = inputs[0] if len(inputs) == 1 else inputs
            if not isinstance(feats, (list, tuple)):
                feats = (feats,)
            captured.append(tuple(feats))

        params = list(self.model.parameters())
        previous = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(True)
        handle = self.head.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                output = self.model(input_tensor)
        finally:
            handle.remove()
            for p, flag in zip(params, previous):
                p.requires_grad_(flag)
        if not captured:
            raise RuntimeError("detection head was never called; check the `head` argument")
        feats = captured[-1]
        for f in feats:
            if torch.is_tensor(f) and f.requires_grad:
                f.retain_grad()
        return _as_pred_tensor(output), feats

    def _class_layout(self, preds):
        """Return ``(channel_index_offset, num_classes, has_objectness)``."""
        channels = int(preds.shape[1])
        nc = self.num_classes
        if nc is None:
            nc = getattr(self.head, "nc", None)
            if isinstance(nc, (list, tuple)):
                nc = nc[0] if nc else None
            nc = int(nc) if nc is not None else None
        if self.class_channel_offset is not None:
            offset = int(self.class_channel_offset)
            has_obj = offset == 5
            if nc is None:
                nc = channels - offset
            return offset, int(nc), has_obj
        if nc is not None and channels == nc + 5:
            return 5, int(nc), True
        if nc is not None and channels == nc + 4:
            return 4, int(nc), False
        # Ambiguous: assume the YOLOv8 layout (box + class scores only).
        return 4, channels - 4, False

    @staticmethod
    def _anchor_grids(feats, input_hw):
        """Per-level ``(cells_y, cells_x, stride)`` for the prediction tensor."""
        grids = []
        for f in feats:
            if not torch.is_tensor(f) or f.ndim != 4:
                continue
            _b, _c, h, w = f.shape
            stride = (input_hw[0] / h + input_hw[1] / w) / 2.0
            grids.append((int(h), int(w), float(stride)))
        return grids

    @staticmethod
    def _box_cells(box_xyxy, grid, level_offset):
        """Grid cells whose centre falls inside ``box_xyxy`` (input coords)."""
        h, w, stride = grid
        ys = (np.arange(h) + 0.5) * stride
        xs = (np.arange(w) + 0.5) * stride
        x1, y1, x2, y2 = box_xyxy
        cols = np.where((xs >= x1) & (xs <= x2))[0]
        rows = np.where((ys >= y1) & (ys <= y2))[0]
        if len(cols) == 0:
            cols = np.array([int(np.clip(round((x1 + x2) / 2.0 / stride - 0.5), 0, w - 1))])
        if len(rows) == 0:
            rows = np.array([int(np.clip(round((y1 + y2) / 2.0 / stride - 0.5), 0, h - 1))])
        flat = (rows[:, None] * w + cols[None, :]).reshape(-1) + level_offset
        return flat, rows, cols

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask
        out = mask.copy()
        h, w = mask.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                src = mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
                dst = out[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
                np.maximum(dst, src, out=dst)
        return out

    # -------------------------------------------------------------- public
    def cam_for_boxes(
        self,
        input_tensor,
        boxes_xyxy: Sequence[Sequence[float]],
        class_ids: Optional[Sequence[int]] = None,
        level: Optional[int] = None,
    ) -> List[np.ndarray]:
        """One CAM per box, in ``input_tensor`` spatial coordinates.

        ``boxes_xyxy`` must already be in the coordinate system of
        ``input_tensor`` (i.e. after the caller's letterbox/resize). A single
        backward pass serves every box: each anchor's score depends only on its
        own (plus a small receptive-field) feature cells, so masking the gradient
        by each box's anchor footprint yields per-detection attribution.
        """
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch required for DetectorGradCAM")
        if len(boxes_xyxy) == 0:
            return []
        input_hw = (int(input_tensor.shape[-2]), int(input_tensor.shape[-1]))
        preds, feats = self._feature_maps(input_tensor)
        if preds.ndim != 3:
            raise ValueError(f"expected (B, C, A) predictions, got {tuple(preds.shape)}")
        offset, _nc, has_obj = self._class_layout(preds)
        grids = self._anchor_grids(feats, input_hw)
        if not grids:
            raise RuntimeError("no 4-D feature maps captured at the detection head")

        boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
        class_ids = (
            [0] * len(boxes) if class_ids is None else [int(c) for c in class_ids]
        )

        # --- choose one cell set per box and build the joint backward score
        per_box_cells = []
        score_terms = []
        level_offsets = np.cumsum([0] + [h * w for h, w, _ in grids])
        for box, cls in zip(boxes, class_ids):
            if level is None:
                area = max(box[2] - box[0], 1.0) * max(box[3] - box[1], 1.0)
                target = float(np.sqrt(area)) / 4.0
                li = int(np.argmin([abs(g[2] - target) for g in grids]))
            else:
                li = int(level)
            cells, rows, cols = self._box_cells(box, grids[li], int(level_offsets[li]))
            per_box_cells.append((li, rows, cols))
            cls_ch = min(int(cls), _nc - 1)
            if has_obj:
                score_terms.append(preds[0, 4, cells].sum() + preds[0, 5 + cls_ch, cells].sum())
            else:
                score_terms.append(preds[0, offset + cls_ch, cells].sum())

        total = sum(score_terms)
        if total.requires_grad:
            total.backward(retain_graph=False)

        cams: List[np.ndarray] = []
        for li, rows, cols in per_box_cells:
            feat = feats[li]
            grad = feat.grad
            if grad is None or not torch.is_tensor(feat) or feat.ndim != 4:
                cams.append(np.zeros((1, 1), dtype=float))
                continue
            h, w = int(feat.shape[-2]), int(feat.shape[-1])
            mask = np.zeros((h, w), dtype=float)
            mask[rows[:, None], cols[None, :]] = 1.0
            mask = self._dilate(mask, self.receptive_radius)
            mask_t = torch.as_tensor(mask, dtype=feat.dtype, device=feat.device)
            f0 = feat[0]
            g0 = grad[0] * mask_t
            weights = g0.mean(dim=(1, 2), keepdim=True)
            cam = F.relu((weights * f0).sum(dim=0))
            if float(cam.detach().max()) <= 0.0:
                # Degenerate (all channel-weighted activations negative, common on
                # near-blank crops): fall back to positive-gradient weighting so the
                # evidence map still localises.
                weights = F.relu(g0).mean(dim=(1, 2), keepdim=True)
                cam = F.relu((weights * f0).sum(dim=0))
            cam = F.interpolate(
                cam[None, None], size=input_hw, mode="bilinear", align_corners=False
            )
            cams.append(cam[0, 0].detach().cpu().numpy())
        return cams
