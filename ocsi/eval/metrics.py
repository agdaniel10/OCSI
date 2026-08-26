"""Quick CLEAR-MOT + IDF1 metrics (paper §6 evaluation).

A pure-numpy/scipy implementation of the headline tracking metrics so ablations
(baseline -> +memory -> +feedback) can be compared *offline, deterministically,
with no external data or the heavier TrackEval dependency*. TrackEval (Phase 6)
remains the reference for HOTA and for official benchmark submission; this module
is the fast inner-loop metric.

Two matching philosophies, both standard:

* **CLEAR-MOT** (Bernardin & Stiefelhagen, 2008) — per-frame IoU matching with
  *sticky* correspondences (prefer to keep the previous frame's pairing), giving
  MOTA, MOTP, precision/recall and the FP / FN / ID-switch counts.
* **IDF1** (Ristani et al., 2016) — one global identity-to-identity assignment
  maximising co-present matched frames, giving IDP / IDR / IDF1.

Inputs are ``{frame: [(track_id, tlwh), ...]}`` dicts (the shape
:func:`ocsi.eval.mot_io.read_gt` returns); :func:`rows_to_frames` adapts the
``(frame, id, x, y, w, h, conf)`` result rows the tracker/pipeline emit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..identity.similarity import iou_matrix

# {frame: [(track_id, tlwh), ...]}
FrameDict = Dict[int, List[Tuple[int, np.ndarray]]]


def _tlwh_to_xyxy_stack(boxes: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a list of ``(4,)`` tlwh boxes into an ``(N, 4)`` xyxy array."""
    arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
    xyxy = arr.copy()
    xyxy[:, 2] = arr[:, 0] + arr[:, 2]
    xyxy[:, 3] = arr[:, 1] + arr[:, 3]
    return xyxy


def rows_to_frames(rows: Sequence[Tuple]) -> FrameDict:
    """Group ``(frame, id, x, y, w, h, conf?)`` result rows into a frame dict."""
    per_frame: FrameDict = {}
    for row in rows:
        frame, tid = int(row[0]), int(row[1])
        tlwh = np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])])
        per_frame.setdefault(frame, []).append((tid, tlwh))
    return per_frame


@dataclass
class MOTMetrics:
    """Aggregate tracking metrics over one sequence (or several, summed)."""

    num_frames: int = 0
    num_gt: int = 0                 # total gt boxes (MOTA denominator)
    num_pred: int = 0               # total predicted boxes
    num_gt_ids: int = 0
    num_pred_ids: int = 0

    tp: int = 0                     # matched pairs (CLEAR-MOT)
    fp: int = 0                     # predictions with no gt match
    fn: int = 0                     # gt boxes with no prediction (misses)
    idsw: int = 0                   # identity switches
    frag: int = 0                   # trajectory fragmentations
    iou_sum: float = 0.0            # sum of matched-pair IoU (for MOTP)

    mostly_tracked: int = 0         # gt ids tracked >= 80% of their life
    partially_tracked: int = 0
    mostly_lost: int = 0            # gt ids tracked <= 20% of their life

    id_tp: int = 0                  # IDF1 true-positive identity frames
    id_fp: int = 0
    id_fn: int = 0

    # ---- derived scores ----
    @property
    def mota(self) -> float:
        """1 - (FN + FP + IDSW) / num_gt. Can be negative for poor trackers."""
        return 1.0 - (self.fn + self.fp + self.idsw) / self.num_gt if self.num_gt else 0.0

    @property
    def motp(self) -> float:
        """Mean IoU over matched pairs (localisation quality; higher is better)."""
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def idp(self) -> float:
        denom = self.id_tp + self.id_fp
        return self.id_tp / denom if denom else 0.0

    @property
    def idr(self) -> float:
        denom = self.id_tp + self.id_fn
        return self.id_tp / denom if denom else 0.0

    @property
    def idf1(self) -> float:
        denom = 2 * self.id_tp + self.id_fp + self.id_fn
        return 2 * self.id_tp / denom if denom else 0.0

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.update(
            mota=self.mota, motp=self.motp, precision=self.precision, recall=self.recall,
            idp=self.idp, idr=self.idr, idf1=self.idf1,
        )
        return d

    def summary(self) -> str:
        return (
            f"MOTA={self.mota:6.3f}  IDF1={self.idf1:6.3f}  MOTP={self.motp:5.3f}  "
            f"IDsw={self.idsw:4d}  FP={self.fp:5d}  FN={self.fn:5d}  "
            f"P={self.precision:5.3f} R={self.recall:5.3f}  "
            f"MT={self.mostly_tracked} PT={self.partially_tracked} ML={self.mostly_lost}"
        )


def _match_frame(
    gt_boxes: List[np.ndarray],
    pred_boxes: List[np.ndarray],
    iou_threshold: float,
    sticky: Dict[int, int],
    gt_ids: List[int],
    pred_ids: List[int],
) -> List[Tuple[int, int, float]]:
    """Match one frame's gt to pred by IoU, preserving the previous frame's pairs.

    ``sticky`` maps ``gt_id -> pred_id`` from the previous frame. Returns a list of
    ``(gt_local_index, pred_local_index, iou)`` matches with IoU >= threshold.
    """
    if not gt_boxes or not pred_boxes:
        return []
    iou = iou_matrix(_tlwh_to_xyxy_stack(gt_boxes), _tlwh_to_xyxy_stack(pred_boxes))

    matches: List[Tuple[int, int, float]] = []
    gt_free = set(range(len(gt_boxes)))
    pred_free = set(range(len(pred_boxes)))

    # 1. keep valid prior correspondences first (suppresses spurious ID switches)
    pred_pos = {pid: j for j, pid in enumerate(pred_ids)}
    for i, gid in enumerate(gt_ids):
        pid = sticky.get(gid)
        if pid is None:
            continue
        j = pred_pos.get(pid)
        if j is not None and j in pred_free and iou[i, j] >= iou_threshold:
            matches.append((i, j, float(iou[i, j])))
            gt_free.discard(i)
            pred_free.discard(j)

    # 2. optimal IoU assignment over what's left, then drop sub-threshold pairs
    if gt_free and pred_free:
        gi = sorted(gt_free)
        pj = sorted(pred_free)
        sub = iou[np.ix_(gi, pj)]
        rows, cols = linear_sum_assignment(-sub)
        for r, c in zip(rows, cols):
            if sub[r, c] >= iou_threshold:
                matches.append((gi[r], pj[c], float(sub[r, c])))
    return matches


def evaluate(gt: FrameDict, pred: FrameDict, iou_threshold: float = 0.5) -> MOTMetrics:
    """Compute CLEAR-MOT + IDF1 metrics for one sequence.

    ``gt`` / ``pred`` are ``{frame: [(id, tlwh), ...]}``. Frames present in only
    one of the two are handled (all-miss or all-FP frames).
    """
    m = MOTMetrics()
    frames = sorted(set(gt) | set(pred))
    m.num_frames = len(frames)

    last_pred_for_gt: Dict[int, int] = {}    # last pred id a gt was EVER matched to
    sticky: Dict[int, int] = {}              # immediate previous frame's pairing
    gt_present: Dict[int, int] = {}          # gt id -> frames present
    gt_tracked: Dict[int, int] = {}          # gt id -> frames matched
    gt_frag_state: Dict[int, bool] = {}      # gt id -> was tracked in its last present frame
    gt_ever_tracked: Dict[int, bool] = {}
    overlap: Dict[Tuple[int, int], int] = {}  # (gt_id, pred_id) -> co-present matched frames

    for f in frames:
        gt_items = gt.get(f, [])
        pred_items = pred.get(f, [])
        gt_ids = [int(t) for t, _ in gt_items]
        pred_ids = [int(t) for t, _ in pred_items]
        gt_boxes = [b for _, b in gt_items]
        pred_boxes = [b for _, b in pred_items]

        m.num_gt += len(gt_items)
        m.num_pred += len(pred_items)
        for gid in gt_ids:
            gt_present[gid] = gt_present.get(gid, 0) + 1

        matches = _match_frame(gt_boxes, pred_boxes, iou_threshold, sticky, gt_ids, pred_ids)

        m.tp += len(matches)
        m.fp += len(pred_items) - len(matches)
        m.fn += len(gt_items) - len(matches)

        matched_gt = set()
        new_sticky: Dict[int, int] = {}
        for i, j, iou_val in matches:
            gid, pid = gt_ids[i], pred_ids[j]
            m.iou_sum += iou_val
            matched_gt.add(gid)
            gt_tracked[gid] = gt_tracked.get(gid, 0) + 1
            overlap[(gid, pid)] = overlap.get((gid, pid), 0) + 1
            new_sticky[gid] = pid
            prev = last_pred_for_gt.get(gid)
            if prev is not None and prev != pid:
                m.idsw += 1
            last_pred_for_gt[gid] = pid

        # fragmentation: a gt that was tracked, present-and-untracked now, tracked again later
        for gid in gt_ids:
            was = gt_frag_state.get(gid, False)
            now = gid in matched_gt
            if was and not now and gt_ever_tracked.get(gid, False):
                m.frag += 1
            gt_frag_state[gid] = now
            if now:
                gt_ever_tracked[gid] = True

        sticky = new_sticky

    _finalize_mt_ml(m, gt_present, gt_tracked)
    _finalize_idf1(m, gt_present, pred, overlap)
    return m


def _finalize_mt_ml(m: MOTMetrics, gt_present: Dict[int, int], gt_tracked: Dict[int, int]) -> None:
    m.num_gt_ids = len(gt_present)
    for gid, total in gt_present.items():
        ratio = gt_tracked.get(gid, 0) / total if total else 0.0
        if ratio >= 0.8:
            m.mostly_tracked += 1
        elif ratio <= 0.2:
            m.mostly_lost += 1
        else:
            m.partially_tracked += 1


def _finalize_idf1(
    m: MOTMetrics,
    gt_present: Dict[int, int],
    pred: FrameDict,
    overlap: Dict[Tuple[int, int], int],
) -> None:
    """Global identity matching: max co-present matched frames, one gt <-> one pred."""
    pred_present: Dict[int, int] = {}
    for items in pred.values():
        for tid, _ in items:
            pred_present[int(tid)] = pred_present.get(int(tid), 0) + 1
    m.num_pred_ids = len(pred_present)

    gt_id_list = sorted(gt_present)
    pred_id_list = sorted(pred_present)
    id_tp = 0
    if gt_id_list and pred_id_list:
        gt_pos = {g: i for i, g in enumerate(gt_id_list)}
        pred_pos = {p: j for j, p in enumerate(pred_id_list)}
        ovl = np.zeros((len(gt_id_list), len(pred_id_list)))
        for (gid, pid), c in overlap.items():
            ovl[gt_pos[gid], pred_pos[pid]] = c
        rows, cols = linear_sum_assignment(-ovl)
        id_tp = int(sum(ovl[r, c] for r, c in zip(rows, cols)))

    m.id_tp = id_tp
    m.id_fn = m.num_gt - id_tp
    m.id_fp = m.num_pred - id_tp
