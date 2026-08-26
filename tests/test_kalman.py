"""Unit tests for the constant-velocity Kalman filter (Phase 3, motion base).

Deterministic, model-free: exercises initiate/predict/update/gating on synthetic
measurements ``[cx, cy, a, h]``.
"""
import numpy as np

from ocsi.motion import KalmanFilter, chi2inv95

M0 = np.array([100.0, 200.0, 0.5, 50.0])  # cx, cy, aspect, height


def test_initiate_shapes_and_zero_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(M0)
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    np.testing.assert_allclose(mean[:4], M0)     # position seeded from measurement
    np.testing.assert_allclose(mean[4:], 0.0)    # velocity unknown -> zero
    assert np.all(np.diag(cov) > 0)              # positive uncertainty everywhere
    np.testing.assert_allclose(cov, cov.T)       # symmetric


def test_predict_advances_position_by_velocity():
    kf = KalmanFilter()
    mean = np.array([100.0, 200.0, 0.5, 50.0, 2.0, -3.0, 0.0, 0.0])
    cov = np.eye(8)
    pred_mean, pred_cov = kf.predict(mean, cov)
    # dt = 1: position += velocity, aspect/height unchanged (their velocities are 0)
    np.testing.assert_allclose(pred_mean[:4], [102.0, 197.0, 0.5, 50.0])
    assert np.trace(pred_cov) > np.trace(cov)    # process noise grows uncertainty


def test_update_pulls_toward_measurement_and_shrinks_covariance():
    kf = KalmanFilter()
    mean, cov = kf.initiate(M0)
    mean, cov = kf.predict(mean, cov)
    trace_prior = np.trace(cov)
    meas = np.array([110.0, 200.0, 0.5, 50.0])   # +10 px in x vs the prediction (100)
    new_mean, new_cov = kf.update(mean, cov, meas)
    assert 100.0 < new_mean[0] < 110.0           # posterior lands between prior and measurement
    assert np.trace(new_cov) < trace_prior       # measurement reduces uncertainty


def test_gating_distance_zero_at_mean():
    kf = KalmanFilter()
    mean, cov = kf.initiate(M0)
    d2 = kf.gating_distance(mean, cov, M0)        # measurement == projected mean
    assert d2.shape == (1,)
    assert d2[0] < 1e-9


def test_gating_distance_grows_with_offset():
    kf = KalmanFilter()
    mean, cov = kf.initiate(M0)
    near = M0 + np.array([5.0, 0.0, 0.0, 0.0])
    far = M0 + np.array([80.0, 0.0, 0.0, 0.0])
    d = kf.gating_distance(mean, cov, np.stack([near, far]))
    assert d.shape == (2,)
    assert d[0] < d[1]


def test_gate_accepts_near_rejects_far():
    kf = KalmanFilter()
    mean, cov = kf.initiate(M0)
    mean, cov = kf.predict(mean, cov)
    proj_center = mean[:4].copy()
    near = proj_center.copy()                     # right on the prediction
    far = proj_center + np.array([500.0, 0.0, 0.0, 0.0])  # ~10 heights away
    d = kf.gating_distance(mean, cov, np.stack([near, far]))
    assert d[0] < chi2inv95[4]                    # 9.4877
    assert d[1] > chi2inv95[4]


def test_converges_on_static_object():
    kf = KalmanFilter()
    mean, cov = kf.initiate(M0)
    for _ in range(30):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, M0)
    np.testing.assert_allclose(mean[:4], M0, atol=0.5)  # settles on the measurement
    assert abs(mean[4]) < 0.5 and abs(mean[5]) < 0.5    # velocity settles near zero
