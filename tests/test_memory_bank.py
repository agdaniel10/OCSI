"""Unit tests for the Object Memory Bank (Phase 2).

These exercise the paper's memory mechanics — confidence-weighted EMA, bounded
queues/gallery, confidence update/decay, the tentative->confirmed->lost->deleted
lifecycle, reactivation, and contamination rollback — with no models or video.
"""
import numpy as np
import pytest

from ocsi.config import MemoryConfig
from ocsi.memory.bank import ObjectMemoryBank
from ocsi.types import Detection, TrackState


def make_det(tlwh=(0, 0, 10, 20), conf=0.9, emb=None, kp=None, frame=0):
    return Detection(
        tlwh=np.array(tlwh, dtype=float),
        confidence=conf,
        embedding=None if emb is None else np.asarray(emb, dtype=float),
        keypoints=kp,
        frame_idx=frame,
    )


def test_create_is_tentative():
    bank = ObjectMemoryBank(MemoryConfig())
    rec = bank.create(make_det(), 0)
    assert rec.state == TrackState.TENTATIVE
    assert rec.hits == 1
    assert len(bank) == 1


def test_confirm_after_min_hits():
    bank = ObjectMemoryBank(MemoryConfig(min_hits=3))
    rec = bank.create(make_det(frame=0), 0)
    bank.update(rec.track_id, make_det(frame=1), reliability=1.0, frame_idx=1)  # hits=2
    assert rec.state == TrackState.TENTATIVE
    bank.update(rec.track_id, make_det(frame=2), reliability=1.0, frame_idx=2)  # hits=3
    assert rec.state == TrackState.CONFIRMED
    assert rec.hits == 3


def test_appearance_ema_confidence_weighted():
    bank = ObjectMemoryBank(MemoryConfig(appearance_ema_alpha=0.9))
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    rec = bank.create(make_det(emb=e1), 0)
    np.testing.assert_allclose(rec.a_bar, e1)  # first observation seeds the prototype
    bank.update(rec.track_id, make_det(emb=e2, frame=1), reliability=1.0, frame_idx=1)
    expected = 0.9 * e1 + 0.1 * e2           # reliability 1.0 -> effective alpha = 0.9
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(rec.a_bar, expected, atol=1e-6)
    assert abs(np.linalg.norm(rec.a_bar) - 1.0) < 1e-6  # stays unit-norm


def test_unreliable_update_preserves_prototype():
    bank = ObjectMemoryBank(MemoryConfig(appearance_ema_alpha=0.9))
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    rec = bank.create(make_det(emb=e1), 0)
    bank.update(rec.track_id, make_det(emb=e2, frame=1), reliability=0.0, frame_idx=1)
    # reliability 0 -> effective alpha = 1.0 -> prototype unchanged
    np.testing.assert_allclose(rec.a_bar, e1, atol=1e-6)


def test_gallery_capacity():
    bank = ObjectMemoryBank(MemoryConfig(gallery_size=5, gallery_reliability_min=0.0))
    rec = bank.create(make_det(emb=[1.0, 0.0, 0.0]), 0)
    for i in range(10):
        v = np.random.RandomState(i).randn(3)
        bank.update(rec.track_id, make_det(emb=v, frame=i + 1), reliability=1.0, frame_idx=i + 1)
    assert len(rec.gallery) == 5


def test_queue_capacity():
    bank = ObjectMemoryBank(MemoryConfig(queue_size=5))
    rec = bank.create(make_det(tlwh=(0, 0, 10, 20)), 0)
    for i in range(10):
        bank.update(rec.track_id, make_det(tlwh=(i, 0, 10, 20), frame=i + 1), reliability=1.0, frame_idx=i + 1)
    assert len(rec.Q_tau) == 5
    assert len(rec.Q_v) == 5


def test_confidence_rises_and_is_bounded():
    cfg = MemoryConfig(confidence_init=0.5, confidence_decay=0.9, confidence_max=1.0)
    bank = ObjectMemoryBank(cfg)
    rec = bank.create(make_det(), 0)
    q0 = rec.q
    bank.update(rec.track_id, make_det(frame=1), reliability=1.0, frame_idx=1)
    assert rec.q > q0
    for i in range(200):
        bank.update(rec.track_id, make_det(frame=i + 2), reliability=1.0, frame_idx=i + 2)
    assert 0.99 < rec.q <= cfg.confidence_max + 1e-9


def test_confidence_decays_on_miss():
    cfg = MemoryConfig(min_hits=2, confidence_init=0.8, confidence_decay=0.9, confidence_min=0.0)
    bank = ObjectMemoryBank(cfg)
    rec = bank.create(make_det(frame=0), 0)
    bank.update(rec.track_id, make_det(frame=1), reliability=1.0, frame_idx=1)  # confirm
    q_before = rec.q
    bank.mark_missed(rec.track_id)
    assert rec.q < q_before
    assert rec.q >= cfg.confidence_min


def test_lifecycle_confirmed_to_lost_to_deleted():
    cfg = MemoryConfig(min_hits=2, max_age=3)
    bank = ObjectMemoryBank(cfg)
    rec = bank.create(make_det(frame=0), 0)
    bank.update(rec.track_id, make_det(frame=1), reliability=1.0, frame_idx=1)  # confirmed
    assert rec.state == TrackState.CONFIRMED

    bank.mark_missed(rec.track_id)          # -> lost, tsu=1
    assert rec.state == TrackState.LOST
    assert bank.step_end() == []            # 1 <= max_age, retained

    for _ in range(3):                      # tsu -> 4 > max_age
        bank.mark_missed(rec.track_id)
    removed = bank.step_end()
    assert rec.track_id in removed
    assert rec.track_id not in bank.records


def test_tentative_dropped_on_miss():
    bank = ObjectMemoryBank(MemoryConfig(min_hits=3))
    rec = bank.create(make_det(), 0)        # tentative
    bank.mark_missed(rec.track_id)
    assert rec.state == TrackState.DELETED
    assert rec.track_id in bank.step_end()
    assert len(bank) == 0


def test_reactivation_from_lost():
    cfg = MemoryConfig(min_hits=2, max_age=10)
    bank = ObjectMemoryBank(cfg)
    rec = bank.create(make_det(frame=0), 0)
    bank.update(rec.track_id, make_det(frame=1), reliability=1.0, frame_idx=1)  # confirmed
    bank.mark_missed(rec.track_id)          # lost
    assert rec.state == TrackState.LOST
    bank.update(rec.track_id, make_det(frame=5), reliability=1.0, frame_idx=5)  # rematched
    assert rec.state == TrackState.CONFIRMED
    assert rec.time_since_update == 0


def test_archive_on_expire():
    cfg = MemoryConfig(min_hits=1, max_age=1, archive_on_expire=True)
    bank = ObjectMemoryBank(cfg)
    rec = bank.create(make_det(frame=0), 0)  # min_hits=1 -> confirm path via update
    bank.update(rec.track_id, make_det(frame=1), reliability=1.0, frame_idx=1)
    bank.mark_missed(rec.track_id)           # lost, tsu=1
    bank.mark_missed(rec.track_id)           # tsu=2 > max_age
    removed = bank.step_end()
    assert rec.track_id in removed
    assert rec.track_id in bank.records      # archived, not deleted
    assert bank.records[rec.track_id].state == TrackState.ARCHIVED


def test_rollback_restores_prototype():
    bank = ObjectMemoryBank(MemoryConfig(appearance_ema_alpha=0.5))
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    rec = bank.create(make_det(emb=e1), 0)
    before = rec.a_bar.copy()
    bank.update(rec.track_id, make_det(emb=e2, frame=1), reliability=1.0, frame_idx=1)
    assert not np.allclose(rec.a_bar, before)
    assert rec.rollback() is True
    np.testing.assert_allclose(rec.a_bar, before)


def test_unique_ids():
    bank = ObjectMemoryBank(MemoryConfig())
    r1 = bank.create(make_det(), 0)
    r2 = bank.create(make_det(), 0)
    assert r1.track_id != r2.track_id
    assert len(bank) == 2
