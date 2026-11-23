import numpy as np

from ocsi.behaviour import TemporalHARRecognizer


def test_multilabel_training_sets_all_positive_class_prototypes():
    rec = TemporalHARRecognizer(feature_dim=4, num_classes=3, min_window=2)
    w1 = np.ones((3, 4), dtype=np.float32)
    w2 = -np.ones((3, 4), dtype=np.float32)

    counts = rec.train_on_multilabel_windows(
        [w1, w2],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
    )

    assert counts == {0: 1, 1: 1, 2: 1}
    for proto in rec._class_protos:
        assert abs(float(np.linalg.norm(proto)) - 1.0) < 1e-5

    obs1 = rec.observe(w1)
    obs2 = rec.observe(w2)
    assert obs1 is not None and obs2 is not None
    assert float(np.dot(rec._class_protos[0], obs1.embedding)) > 0.99
    assert float(np.dot(rec._class_protos[1], obs2.embedding)) > 0.99
