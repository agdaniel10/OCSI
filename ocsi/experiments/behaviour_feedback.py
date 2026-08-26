"""Synthetic behaviour-feedback experiment for OCSI contribution #3.

This is a deliberately small, model-free experiment around the exact association
math used by the tracker:

* confidence gate: low-confidence activity windows should have no effect;
* behaviour blend: reliable activity can rescue an otherwise ambiguous match;
* contradiction rejection: reliable anti-correlated behaviour can veto a pair.

Run::

    python -m ocsi.experiments.behaviour_feedback
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..behaviour import behaviour_gate, blend_behaviour, contradiction_mask
from ..config import OCSIConfig
from ..identity import solve_assignment
from ..identity.similarity import cos_to_01, cosine_matrix


@dataclass
class ScenarioResult:
    name: str
    expected: List[Tuple[int, int]]
    matches: List[Tuple[int, int]]
    passed: bool
    gate: List[List[float]]
    score: List[List[float]]
    valid: List[List[bool]]


def _assignment(
    base_score: np.ndarray,
    track_behaviour: np.ndarray,
    det_behaviour: np.ndarray,
    p_max: np.ndarray,
    cfg: OCSIConfig,
    *,
    feedback_enabled: bool,
    gate_enabled: bool,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(base_score, dtype=float).copy()
    valid = np.ones_like(score, dtype=bool)
    gate = np.zeros_like(score, dtype=float)

    if feedback_enabled:
        cos_bt = cosine_matrix(track_behaviour, det_behaviour)
        gate = behaviour_gate(
            p_max.reshape(1, -1),
            np.zeros((len(track_behaviour), 1)),
            cfg.behaviour.theta_b,
            cfg.behaviour.tau_b,
            gate_enabled=gate_enabled,
        )
        score = blend_behaviour(score, cos_to_01(cos_bt), gate, cfg.association.w_behaviour)
        valid &= ~contradiction_mask(
            cos_bt,
            gate > 0.0,
            cfg.behaviour.contradiction_threshold,
        )

    matches, _, _ = solve_assignment(score, valid, cfg.association.s_min)
    return sorted(matches), gate, score, valid


def run(save_json: Optional[str] = None) -> Dict:
    """Run two deterministic association scenarios and print a compact report."""
    cfg = OCSIConfig()
    cfg.association.w_behaviour = 1.0
    cfg.association.s_min = 0.0
    cfg.behaviour.theta_b = 0.60
    cfg.behaviour.contradiction_threshold = -0.20

    track_behaviour = np.array([[1.0, 0.0], [-1.0, 0.0]])
    behaviour_diagonal = np.array([[1.0, 0.0], [-1.0, 0.0]])
    behaviour_cross = np.array([[-1.0, 0.0], [1.0, 0.0]])

    cases = [
        (
            "reliable behaviour fixes ambiguous base association",
            np.array([[0.62, 0.66], [0.66, 0.62]]),
            behaviour_diagonal,
            np.array([0.92, 0.92]),
            True,
            True,
            [(0, 0), (1, 1)],
        ),
        (
            "low-confidence behaviour is ignored by the gate",
            np.array([[0.66, 0.62], [0.62, 0.66]]),
            behaviour_cross,
            np.array([0.40, 0.40]),
            True,
            True,
            [(0, 0), (1, 1)],
        ),
        (
            "ungated low-confidence behaviour corrupts association",
            np.array([[0.66, 0.62], [0.62, 0.66]]),
            behaviour_cross,
            np.array([0.40, 0.40]),
            True,
            False,
            [(0, 1), (1, 0)],
        ),
    ]

    results: List[ScenarioResult] = []
    for name, base, det_behaviour, p_max, feedback, gate_enabled, expected in cases:
        matches, gate, score, valid = _assignment(
            base,
            track_behaviour,
            det_behaviour,
            p_max,
            cfg,
            feedback_enabled=feedback,
            gate_enabled=gate_enabled,
        )
        results.append(
            ScenarioResult(
                name=name,
                expected=expected,
                matches=matches,
                passed=matches == expected,
                gate=np.round(gate, 3).tolist(),
                score=np.round(score, 3).tolist(),
                valid=valid.tolist(),
            )
        )

    _print_report(results)
    payload = {
        "config": {
            "theta_b": cfg.behaviour.theta_b,
            "w_behaviour": cfg.association.w_behaviour,
            "contradiction_threshold": cfg.behaviour.contradiction_threshold,
        },
        "scenarios": [asdict(r) for r in results],
        "all_passed": all(r.passed for r in results),
    }
    if save_json:
        with open(save_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nsaved: {save_json}")
    return payload


def _print_report(results: List[ScenarioResult]) -> None:
    print("\n=== Behaviour feedback association experiment ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {status}  {r.name}")
        print(f"        matches={r.matches} expected={r.expected}")
        print(f"        gate={r.gate}")
        print(f"        valid={r.valid}")
        print(f"        score={r.score}")


if __name__ == "__main__":  # pragma: no cover
    run(save_json="behaviour_feedback_results.json")
