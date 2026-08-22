"""Unit tests for the detector adapter (Phase 1).

The Ultralytics-backed :class:`YOLODetector` needs the optional ``perception`` extra
and downloaded weights, so it is not exercised here; the pure post-processing
(:func:`filter_detections`), which is where the OCSI-specific logic lives, is.
"""
import numpy as np

from ocsi.perception import filter_detections
from ocsi.types import Detection


def test_keeps_only_person_class_above_threshold():
    boxes = np.array([
        [0.0, 0.0, 10.0, 20.0],     # person, high conf   -> keep
        [5.0, 5.0, 10.0, 20.0],     # person, low conf    -> drop
        [9.0, 9.0, 10.0, 20.0],     # car (class 2)       -> drop
    ])
    scores = [0.9, 0.2, 0.99]
    classes = [0, 0, 2]
    dets = filter_detections(boxes, scores, classes, person_class_id=0, conf_threshold=0.5, frame_idx=4)
    assert len(dets) == 1
    d = dets[0]
    assert isinstance(d, Detection)
    assert d.confidence == 0.9 and d.class_id == 0 and d.frame_idx == 4
    np.testing.assert_allclose(d.tlwh, [0.0, 0.0, 10.0, 20.0])


def test_empty_input_returns_empty():
    assert filter_detections(np.zeros((0, 4)), [], [], 0, 0.5) == []


def test_non_person_class_configurable():
    boxes = np.array([[0.0, 0.0, 5.0, 5.0]])
    dets = filter_detections(boxes, [0.8], [3], person_class_id=3, conf_threshold=0.5)
    assert len(dets) == 1 and dets[0].class_id == 3
