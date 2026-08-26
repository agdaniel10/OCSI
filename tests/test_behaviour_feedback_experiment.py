"""Tests for the behaviour-feedback experiment.

The experiment is deliberately model-free, so these assertions lock the observable
association outcomes that demonstrate the confidence gate's value.
"""

from ocsi.experiments.behaviour_feedback import run


def test_behaviour_feedback_experiment_passes_all_scenarios():
    payload = run()

    assert payload["all_passed"] is True
    scenarios = {s["name"]: s for s in payload["scenarios"]}

    assert scenarios[
        "reliable behaviour fixes ambiguous base association"
    ]["matches"] == [(0, 0), (1, 1)]
    assert scenarios[
        "low-confidence behaviour is ignored by the gate"
    ]["matches"] == [(0, 0), (1, 1)]
    assert scenarios[
        "ungated low-confidence behaviour corrupts association"
    ]["matches"] == [(0, 1), (1, 0)]


def test_low_confidence_gated_scenario_leaves_scores_unchanged():
    payload = run()
    scenario = {
        s["name"]: s for s in payload["scenarios"]
    }["low-confidence behaviour is ignored by the gate"]

    assert scenario["gate"] == [[0.0, 0.0], [0.0, 0.0]]
    assert scenario["score"] == [[0.66, 0.62], [0.62, 0.66]]
    assert scenario["valid"] == [[True, True], [True, True]]
