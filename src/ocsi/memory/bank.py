"""The Object Memory Bank: a track-indexed key-value store of identity records
(paper §3.3). Owns creation, confirmation, lost-state retention, decay,
deletion/archiving and contamination rollback (paper §3.4).
"""
from __future__ import annotations

from typing import Dict, List

from ..config import MemoryConfig
from ..types import Detection, TrackState
from .record import MemoryRecord


class ObjectMemoryBank:
    """Bounded, confidence-aware memory keyed by track id."""

    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.records: Dict[int, MemoryRecord] = {}
        self._next_id = 1

    # ------------------------------------------------------------- creation
    def create(self, detection: Detection, frame_idx: int) -> MemoryRecord:
        tid = self._next_id
        self._next_id += 1
        rec = MemoryRecord(tid, detection, self.cfg, frame_idx)
        self.records[tid] = rec
        return rec

    # -------------------------------------------------------- per-frame ops
    def update(self, track_id: int, detection: Detection, reliability: float, frame_idx: int) -> MemoryRecord:
        """Confidence-gated update of a matched track. Confirms tentative tracks
        once they reach ``min_hits``."""
        rec = self.records[track_id]
        rec.mark_matched(detection, reliability, frame_idx)
        if rec.state == TrackState.TENTATIVE and rec.hits >= self.cfg.min_hits:
            rec.state = TrackState.CONFIRMED
        return rec

    def update_behaviour(self, track_id: int, probs, embedding, reliability: float) -> None:
        """Forward a reliable behaviour observation to a record's behaviour-prototype
        EMA (paper §3.5 step 13). The tracker calls this only when the window's activity
        confidence clears ``theta_b``; it resets the record's ``frames_since_reliable``
        staleness counter, which the confidence gate decays over."""
        self.records[track_id].update_behaviour(probs, embedding, reliability)

    def mark_missed(self, track_id: int) -> None:
        """A track that found no detection this frame: unconfirmed tracks are
        dropped; confirmed tracks move to the lost state for later reactivation."""
        rec = self.records[track_id]
        rec.mark_missed()
        if rec.state == TrackState.TENTATIVE:
            rec.state = TrackState.DELETED
        elif rec.state == TrackState.CONFIRMED:
            rec.state = TrackState.LOST

    def step_end(self, frame_idx: int = -1) -> List[int]:
        """Retire expired records at the end of a frame. Returns removed ids."""
        removed: List[int] = []
        for tid, rec in list(self.records.items()):
            expired_lost = rec.state == TrackState.LOST and rec.time_since_update > self.cfg.max_age
            if rec.state == TrackState.DELETED or expired_lost:
                if self.cfg.archive_on_expire and expired_lost:
                    rec.state = TrackState.ARCHIVED  # kept out of matching; future cross-camera
                else:
                    del self.records[tid]
                removed.append(tid)
        return removed

    # ---------------------------------------------------------- retrieval
    def matchable(self) -> List[MemoryRecord]:
        """Records eligible for detection association: active + retained lost."""
        return [r for r in self.records.values() if r.is_matchable]

    def active(self) -> List[MemoryRecord]:
        return [
            r for r in self.records.values()
            if r.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)
        ]

    def lost(self) -> List[MemoryRecord]:
        return [r for r in self.records.values() if r.state == TrackState.LOST]

    def confirmed(self) -> List[MemoryRecord]:
        return [r for r in self.records.values() if r.state == TrackState.CONFIRMED]

    def get(self, track_id: int) -> MemoryRecord:
        return self.records[track_id]

    def __len__(self) -> int:
        return len(self.records)
