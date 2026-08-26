"""Synthetic contamination-rollback experiment for the Object Memory Bank.

The paper claims the memory record can recover from a bad identity update by
rolling back to a saved clean state. This model-free experiment measures that
directly: apply an intentionally wrong appearance/behaviour update, detect the
conflict, then compare memory quality with and without rollback.

Run::

    python -m ocsi.experiments.contamination_rollback
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from ..config import MemoryConfig
from ..identity.similarity import cosine_matrix
from ..memory import ObjectMemoryBank
from ..types import Detection


@dataclass
class RollbackResult:
    conflict_detected: bool
    rollback_applied: bool
    clean_appearance_cosine: float
    contaminated_appearance_cosine: float
    recovered_appearance_cosine: float
    contaminated_behaviour_cosine: float
    recovered_behaviour_cosine: float
    clean_confidence: float
    contaminated_confidence: float
    recovered_confidence: float
    passed: bool


def _det(frame: int, embedding: np.ndarray) -> Detection:
    return Detection(
        tlwh=np.array([100.0 + frame, 80.0, 36.0, 72.0]),
        confidence=0.95,
        class_id=0,
        frame_idx=frame,
        embedding=embedding,
    )


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_matrix(np.asarray([a]), np.asarray([b]))[0, 0])


def run(save_json: Optional[str] = None) -> Dict:
    cfg = MemoryConfig(
        appearance_ema_alpha=0.2,
        confidence_init=0.6,
        confidence_decay=0.2,
        min_hits=2,
    )
    bank = ObjectMemoryBank(cfg)

    true_app = np.array([1.0, 0.0, 0.0])
    wrong_app = np.array([0.0, 1.0, 0.0])
    true_beh = np.array([0.0, 1.0, 0.0])
    wrong_beh = np.array([0.0, -1.0, 0.0])
    true_probs = np.array([0.95, 0.03, 0.02])
    wrong_probs = np.array([0.02, 0.95, 0.03])

    rec = bank.create(_det(0, true_app), 0)
    bank.update(rec.track_id, _det(1, true_app), reliability=1.0, frame_idx=1)
    bank.update_behaviour(rec.track_id, true_probs, true_beh, reliability=1.0)

    clean_app = rec.a_bar.copy()
    clean_beh = rec.b.copy()
    clean_q = rec.q

    bank.update(rec.track_id, _det(2, wrong_app), reliability=1.0, frame_idx=2)
    bank.update_behaviour(rec.track_id, wrong_probs, wrong_beh, reliability=1.0)

    contaminated_app = rec.a_bar.copy()
    contaminated_beh = rec.b.copy()
    contaminated_q = rec.q
    app_conflict = _cos(clean_app, wrong_app) < 0.25
    behaviour_conflict = _cos(clean_beh, wrong_beh) < -0.20
    conflict_detected = bool(app_conflict or behaviour_conflict)
    rollback_applied = rec.rollback() if conflict_detected else False

    result = RollbackResult(
        conflict_detected=conflict_detected,
        rollback_applied=rollback_applied,
        clean_appearance_cosine=_cos(clean_app, true_app),
        contaminated_appearance_cosine=_cos(contaminated_app, true_app),
        recovered_appearance_cosine=_cos(rec.a_bar, true_app),
        contaminated_behaviour_cosine=_cos(contaminated_beh, true_beh),
        recovered_behaviour_cosine=_cos(rec.b, true_beh),
        clean_confidence=round(clean_q, 6),
        contaminated_confidence=round(contaminated_q, 6),
        recovered_confidence=round(rec.q, 6),
        passed=bool(
            conflict_detected
            and rollback_applied
            and _cos(contaminated_app, true_app) < 0.5
            and _cos(rec.a_bar, true_app) > 0.999
            and _cos(contaminated_beh, true_beh) < 0.0
            and _cos(rec.b, true_beh) > 0.999
            and abs(rec.q - clean_q) < 1e-9
        ),
    )

    _print_report(result)
    payload = {"result": asdict(result), "all_passed": result.passed}
    if save_json:
        with open(save_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nsaved: {save_json}")
    return payload


def _print_report(result: RollbackResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print("\n=== Memory contamination rollback experiment ===")
    print(f"  {status} conflict_detected={result.conflict_detected} rollback={result.rollback_applied}")
    print(
        "  appearance cosine to clean identity: "
        f"clean={result.clean_appearance_cosine:.3f} "
        f"contaminated={result.contaminated_appearance_cosine:.3f} "
        f"recovered={result.recovered_appearance_cosine:.3f}"
    )
    print(
        "  behaviour cosine to clean identity: "
        f"contaminated={result.contaminated_behaviour_cosine:.3f} "
        f"recovered={result.recovered_behaviour_cosine:.3f}"
    )
    print(
        "  confidence: "
        f"clean={result.clean_confidence:.3f} "
        f"contaminated={result.contaminated_confidence:.3f} "
        f"recovered={result.recovered_confidence:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    run(save_json="contamination_rollback_results.json")
