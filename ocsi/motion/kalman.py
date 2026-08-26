"""Constant-velocity Kalman filter over bounding-box state (paper §4: the
"Kalman/ByteTrack-style base we extend").

State is 8-dimensional ``[cx, cy, a, h, vx, vy, va, vh]`` — box center, aspect
ratio ``a = w/h``, height ``h`` and their velocities. The measurement is the
4-vector ``[cx, cy, a, h]`` (``Detection.xyah``). This is the standard
SORT/DeepSORT formulation; OCSI reuses it unchanged and layers the memory-aware
association on top, so this module stays a self-contained, model-free numpy unit.

The filter is *stateless*: ``(mean, covariance)`` are passed in and returned, so
each :class:`~ocsi.memory.record.MemoryRecord` can own its own track state.

Motion contributes to association two ways (both via :meth:`gating_distance`,
which returns the squared Mahalanobis distance in measurement space):
  * ``S_motion = exp(-0.5 * d^2)`` — a [0,1] motion-similarity term.
  * a chi-square gate: reject det-track pairs with ``d^2`` above
    ``chi2inv95[4] = 9.4877`` (``AssociationConfig.mahalanobis_gate``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg

# 0.95 quantile of the chi-square distribution by degrees of freedom, for gating.
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """A constant-velocity Kalman filter for image-space bounding boxes.

    Observed directly: the box center ``(cx, cy)``, aspect ratio ``a`` and
    height ``h``. Their velocities are estimated (constant-velocity model). The
    process/observation noise standard deviations are scaled by the current
    height ``h``, so uncertainty is proportional to object scale.
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # State transition F (position += velocity * dt) and observation H.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Noise scales relative to object height (DeepSORT defaults).
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    # ------------------------------------------------------------------ init
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a track state from an unassociated measurement ``[cx, cy, a, h]``.

        Velocities are initialised to zero with high covariance (unknown until a
        second observation constrains them).
        """
        measurement = np.asarray(measurement, dtype=float)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    # --------------------------------------------------------------- predict
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate the state one frame forward (prior)."""
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    # --------------------------------------------------------------- project
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space ``[cx, cy, a, h]``."""
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    # ---------------------------------------------------------------- update
    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct the predicted state with an observed measurement (posterior)."""
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain via Cholesky solve (numerically stabler than an explicit inverse).
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = np.asarray(measurement, dtype=float) - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    # ------------------------------------------------------- gating distance
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Squared Mahalanobis distance from the state to each measurement.

        ``measurements`` is an ``(N, 4)`` array of ``[cx, cy, a, h]`` rows;
        returns an ``(N,)`` array of squared distances. Compare against
        ``chi2inv95`` to gate, or feed into ``exp(-0.5 * d^2)`` for a motion
        similarity in [0,1].
        """
        mean, covariance = self.project(mean, covariance)
        measurements = np.atleast_2d(np.asarray(measurements, dtype=float))
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(z * z, axis=0)
