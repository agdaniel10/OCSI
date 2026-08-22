"""Unit tests for the detector adapter (Phase 1).

The Ultralytics-backed :class:`YOLODetector` needs the optional ``perception`` extra
and downloaded weights, so it is not exercised here; the pure post-processing
(:func:`filter_detections`), which is where the OCSI-specific logic lives, is.
"""
import numpy as np

from ocsi.config import PerceptionConfig
from ocsi.perception import filter_detections
from ocsi.perception.detector import YOLODetector
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


def test_yolo_detector_resolves_auto_device(monkeypatch):
    class _Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    class _Boxes:
        xyxy = _Tensor([[1.0, 2.0, 11.0, 22.0]])
        conf = _Tensor([0.9])
        cls = _Tensor([0])

        def __len__(self):
            return 1

    class _Result:
        boxes = _Boxes()

    class _Model:
        def __init__(self):
            self.kwargs = None

        def predict(self, *args, **kwargs):
            self.kwargs = kwargs
            return [_Result()]

    model = _Model()

    class _YOLO:
        def __init__(self, weights):
            self.weights = weights

        def predict(self, *args, **kwargs):
            return model.predict(*args, **kwargs)

    monkeypatch.setattr("ocsi.perception.detector.resolve_device", lambda device: "cuda")
    import sys
    import types

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_YOLO))

    detector = YOLODetector(PerceptionConfig(device="auto"))
    dets = detector(np.zeros((24, 24, 3), dtype=np.uint8))

    assert detector.device == "cuda"
    assert model.kwargs["device"] == "cuda"
    assert len(dets) == 1
