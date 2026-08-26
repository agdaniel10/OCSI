"""Unit tests for the feature cache (Phase 1)."""
import numpy as np

from ocsi.perception import FeatureCache


def test_in_memory_put_get_has():
    c = FeatureCache(root=None)
    assert not c.has("k")
    assert c.get("k") is None
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    c.put("k", arr)
    assert c.has("k")
    np.testing.assert_array_equal(c.get("k"), arr)


def test_disk_persists_across_instances(tmp_path):
    root = str(tmp_path / "cache")
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    FeatureCache(root).put("seq/000001", arr)          # nested key -> subdir
    fresh = FeatureCache(root)                           # new process/instance
    assert fresh.has("seq/000001")
    np.testing.assert_array_equal(fresh.get("seq/000001"), arr)


def test_get_or_compute_computes_once(tmp_path):
    c = FeatureCache(str(tmp_path / "c"))
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.ones((3,), dtype=np.float32)

    a = c.get_or_compute("x", compute)
    b = c.get_or_compute("x", compute)                   # served from cache
    np.testing.assert_array_equal(a, b)
    assert calls["n"] == 1


def test_memory_disabled_still_uses_disk(tmp_path):
    root = str(tmp_path / "c")
    c = FeatureCache(root, memory=False)
    c.put("k", np.zeros((2,), dtype=np.float32))
    assert c.has("k")                                    # from disk, not memory
    np.testing.assert_array_equal(c.get("k"), np.zeros((2,)))
