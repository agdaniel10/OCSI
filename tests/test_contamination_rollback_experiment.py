"""Tests for the contamination-rollback experiment."""

from ocsi.experiments.contamination_rollback import run


def test_contamination_rollback_experiment_restores_clean_memory():
    payload = run()
    result = payload["result"]

    assert payload["all_passed"] is True
    assert result["conflict_detected"] is True
    assert result["rollback_applied"] is True

    assert result["contaminated_appearance_cosine"] < 0.5
    assert result["recovered_appearance_cosine"] > 0.999
    assert result["contaminated_behaviour_cosine"] < 0.0
    assert result["recovered_behaviour_cosine"] > 0.999


def test_contamination_rollback_restores_confidence_snapshot():
    result = run()["result"]

    assert result["contaminated_confidence"] > result["clean_confidence"]
    assert result["recovered_confidence"] == result["clean_confidence"]
