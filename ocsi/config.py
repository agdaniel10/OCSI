"""Configuration for OCSI.

All thresholds, weights and capacities live here as dataclasses with documented
defaults. These are OUR engineering choices — the source paper leaves the
corresponding hyperparameter table blank ("[AUTHOR TO COMPLETE]"). ``configs/
default.yaml`` mirrors these values; :meth:`OCSIConfig.from_yaml` overrides them.

Note: this module intentionally does NOT use ``from __future__ import annotations``
so that dataclass field types stay as real objects and ``_build_dataclass`` can
introspect nested dataclasses for YAML/dict loading.
"""
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Tuple


@dataclass
class MemoryConfig:
    """Object Memory Bank capacities and update/decay rules (paper §3.3–3.4)."""

    appearance_ema_alpha: float = 0.9   # base (minimum) EMA weight on the prototype
    gallery_size: int = 30              # |G| bounded appearance gallery
    queue_size: int = 30                # |Q_*| trajectory/velocity/pose/context queues
    confidence_init: float = 0.5        # q at creation
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    confidence_decay: float = 0.9       # delta on a reliable match: q <- clip(delta*q + (1-delta)*r)
    confidence_decay_miss: float = 0.98  # per-missed-frame multiplicative decay (slow, so lost
    #                                      tracks retain enough confidence to be reactivated)
    gallery_reliability_min: float = 0.5  # only store embeddings at/above this reliability
    min_hits: int = 3                   # hits before tentative -> confirmed
    max_age: int = 30                   # frames a lost track is retained for reactivation
    rollback_history: int = 3           # snapshots kept for contamination rollback
    archive_on_expire: bool = False     # archive (vs delete) expired records; cross-camera = future


@dataclass
class ContaminationConfig:
    """Contamination-rollback control (paper §3.4 step 14).

    The tracker detects a likely incorrect match by comparing the detection
    embedding against the track's appearance prototype. When the cosine is below
    ``appearance_conflict_threshold`` AND the track's memory confidence is at or
    above ``confidence_min``, the match is flagged. After ``confirm_frames``
    consecutive conflicts, the record is rolled back to its last clean snapshot.
    """

    enabled: bool = True                # master switch for contamination detection
    appearance_conflict_threshold: float = 0.30  # raw cosine below this = likely wrong identity
    confidence_min: float = 0.70        # only flag conflicts on high-confidence tracks
    confirm_frames: int = 2             # consecutive conflicts before rollback is applied


@dataclass
class AssociationConfig:
    """Unified association score, gating and assignment (paper §4.2–4.3)."""

    # default softmax-normalized cue weights (behaviour handled separately via the gate)
    w_app: float = 0.30
    w_motion: float = 0.20
    w_iou: float = 0.20
    w_memory: float = 0.20
    w_behaviour: float = 0.10           # only contributes when behaviour feedback is enabled
    w_pose: float = 0.05                # pose-similarity weight (used when pose features exist)
    w_context: float = 0.05             # context-similarity weight (used when context features exist)
    use_memory: bool = True             # ablation: False -> baseline (short-term appearance only)
    s_min: float = 0.30                 # reject assignments scoring below this
    iou_gate: float = 0.10              # minimum geometric plausibility
    mahalanobis_gate: float = 9.4877    # chi-square 0.95 quantile, 4 dof (Kalman gating)
    two_stage: bool = True              # ByteTrack-style high- then low-confidence matching
    high_conf_threshold: float = 0.50   # split high/low detection confidence
    reactivation: bool = True           # allow LOST tracks to re-associate by appearance after
    #                                     occlusion, bypassing the IoU floor (the memory bank's
    #                                     core value; paper §3.4 reactivation step). Ablated OFF
    #                                     in the 'baseline' stage so recovery gain is attributable.
    reactivation_app_gate: float = 0.84  # min RAW cosine (-1..1) of a detection vs a lost track's
    #                                      gallery to permit appearance-only reactivation; tuned
    #                                      above MOT17 diff-ID ranges observed with the default
    #                                      ResNet-18 features, not below them
    adaptive_reactivation: bool = False   # estimate a sequence-specific reactivation gate from
    #                                      held-out/cached embedding diagnostics instead of using
    #                                      the fixed global gate
    adaptive_diff_margin: float = 0.03    # keep the gate above observed different-ID similarity
    adaptive_same_margin: float = 0.02    # keep the gate below observed same-ID similarity when
    #                                      the embedding separation allows it
    adaptive_weights: bool = False        # paper §4.3: modulate cue weights by reliability
    #                                      (occlusion ratio, motion uncertainty, behaviour conf)


@dataclass
class BehaviourConfig:
    """Behaviour recognition + confidence gate (paper §3.5, §4)."""

    enabled: bool = False               # ablation flag: closed-loop feedback on/off
    gate_enabled: bool = True           # apply the confidence gate g; False -> g≡1 (naive
    #                                     full-trust HAR, the 'ungated' strawman for the experiment)
    theta_b: float = 0.60               # activity-confidence threshold in the gate
    tau_b: float = 30.0                 # gate temporal-decay constant (frames)
    window: int = 16                    # temporal window length L for HAR
    min_window: int = 8                 # minimum frames before HAR is attempted
    contradiction_threshold: float = -0.20  # cos(b_det, b_track) below which a match is vetoed
    embedding_dim: int = 64             # behaviour embedding dimension
    num_classes: int = 4                # default activity set size (stub/light model)


@dataclass
class PerceptionConfig:
    """Detector / Re-ID / pose adapters (Phase 1)."""

    detector_weights: str = "yolov8s.pt"
    person_class_id: int = 0
    det_conf_threshold: float = 0.15
    reid_backend: str = "auto"          # "auto" (torchreid/OSNet -> resnet50 -> resnet18),
    #                                     "torchvision" control, or "torchreid" person-ReID
    reid_backbone: str = "resnet18"     # torchvision backbone; OSNet is an optional upgrade
    reid_input_hw: Tuple[int, int] = (256, 128)  # (height, width) standard Re-ID crop
    reid_pretrained: bool = True        # load ImageNet weights (needs a one-off download);
    #                                     falls back to random init offline, with a warning
    reid_batch_size: int = 64           # crops per forward pass
    device: str = "auto"                # "auto" -> cuda when available, else cpu
    use_pose: bool = False              # off by default on CPU
    pose_weights: str = "yolov8n-pose.pt"


@dataclass
class OCSIConfig:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    behaviour: BehaviourConfig = field(default_factory=BehaviourConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    # ---- (de)serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCSIConfig":
        return _build_dataclass(cls, data or {})

    @classmethod
    def from_yaml(cls, path: str) -> "OCSIConfig":
        import yaml  # optional dependency; only needed if YAML configs are used

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)


def _build_dataclass(dc_type, data):
    """Recursively construct a (possibly nested) dataclass from a dict, keeping
    defaults for any keys not present."""
    if not is_dataclass(dc_type):
        return data
    kwargs = {}
    type_hints = {f.name: f.type for f in fields(dc_type)}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = type_hints[f.name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build_dataclass(ftype, value)
        elif isinstance(value, list):
            kwargs[f.name] = tuple(value) if "Tuple" in str(ftype) else value
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)


def cuda_available() -> bool:
    """Return True when PyTorch can see a CUDA device.

    Kept as a tiny helper so config/device tests can monkeypatch it without
    importing heavyweight perception adapters.
    """
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(device: str = "auto") -> str:
    """Resolve a configured device string to a usable torch/Ultralytics device.

    ``"auto"`` prefers CUDA when PyTorch reports it available, otherwise CPU.
    Explicit CUDA requests fall back to CPU with a warning instead of crashing
    late inside model construction on machines without a CUDA runtime.
    """
    requested = (device or "auto").strip().lower()
    if requested in ("auto", "gpu"):
        return "cuda" if cuda_available() else "cpu"
    if requested.startswith("cuda") and not cuda_available():
        warnings.warn(
            f"requested device {device!r}, but CUDA is not available; falling back to CPU",
            RuntimeWarning,
        )
        return "cpu"
    return requested


# --- ablation presets: baseline -> +memory -> +behaviour feedback (paper §6.4, Table 7) ---

ABLATION_STAGES = ("baseline", "memory", "adaptive_memory", "feedback")


def apply_ablation(config: OCSIConfig, stage: str) -> OCSIConfig:
    """Return a copy of ``config`` reconfigured for an ablation stage.

    - ``baseline``  : short-term appearance only, no memory term, no behaviour.
    - ``memory``    : add the Object Memory Bank memory-similarity term.
    - ``adaptive_memory`` : memory with a data-calibrated reactivation gate.
    - ``feedback``  : add confidence-gated behaviour feedback (full OCSI).
    """
    if stage not in ABLATION_STAGES:
        raise ValueError(f"unknown ablation stage {stage!r}; expected one of {ABLATION_STAGES}")
    cfg = OCSIConfig.from_dict(config.to_dict())  # deep copy via round-trip
    if stage == "baseline":
        cfg.association.use_memory = False
        cfg.association.reactivation = False   # no appearance rescue: motion/IoU association only
        cfg.association.adaptive_reactivation = False
        cfg.association.adaptive_weights = False
        cfg.behaviour.enabled = False
        cfg.contamination.enabled = False
    elif stage == "memory":
        cfg.association.use_memory = True
        cfg.association.reactivation = True    # memory bank persists identity across occlusion
        cfg.association.adaptive_reactivation = False
        cfg.association.adaptive_weights = False
        cfg.behaviour.enabled = False
        cfg.contamination.enabled = True
    elif stage == "adaptive_memory":
        cfg.association.use_memory = True
        cfg.association.reactivation = True
        cfg.association.adaptive_reactivation = True
        cfg.association.adaptive_weights = False
        cfg.behaviour.enabled = False
        cfg.contamination.enabled = True
    elif stage == "feedback":
        cfg.association.use_memory = True
        cfg.association.reactivation = True
        cfg.association.adaptive_reactivation = False
        cfg.association.adaptive_weights = True
        cfg.behaviour.enabled = True
        cfg.contamination.enabled = True
    return cfg