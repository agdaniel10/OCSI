"""Tests for the Kaggle runner helpers (no Kaggle, no MOT data needed)."""
import os

import kaggle_xocsi


def _make_seq(root, name):
    seq = os.path.join(root, name)
    os.makedirs(os.path.join(seq, "img1"))
    os.makedirs(os.path.join(seq, "gt"))
    with open(os.path.join(seq, "gt", "gt.txt"), "w", encoding="utf-8"):
        pass
    return seq


def test_find_sequences_dedupes_mot17_variants(tmp_path):
    root = str(tmp_path)
    _make_seq(root, "MOT17-02-DPM")
    _make_seq(root, "MOT17-02-FRCNN")
    _make_seq(root, "MOT17-04-FRCNN")
    # A sequence without ground truth (the test split) is not usable for evaluation.
    os.makedirs(os.path.join(root, "MOT17-10-FRCNN", "img1"))

    found = [os.path.basename(p) for p in kaggle_xocsi.find_sequences(root)]

    assert found == ["MOT17-02-FRCNN", "MOT17-04-FRCNN"]


def test_stage_rows_surfaces_cecs_and_coverage():
    payload = {
        "runs": [
            {
                "sequence": "MOT17-02-FRCNN",
                "seed": 0,
                "attribution": {"coverage": 1.0},
                "results": [
                    {
                        "stage": "x-ocsi",
                        "metrics": {
                            "idf1": 0.5,
                            "mota": 0.4,
                            "idsw": 3,
                            "cecs": {
                                "num_cecs": 7,
                                "cecs_mean": 0.6,
                                "cecs_min": 0.2,
                                "cecs_max": 0.9,
                            },
                        },
                    },
                    {
                        "stage": "baseline",
                        "metrics": {"idf1": 0.4, "mota": 0.3, "idsw": 5},
                    },
                ],
            }
        ]
    }

    rows = kaggle_xocsi.stage_rows(payload)

    assert [r["stage"] for r in rows] == ["x-ocsi", "baseline"]
    assert rows[0]["cecs_num"] == 7
    assert rows[0]["attribution_coverage"] == 1.0
    assert rows[1]["cecs_num"] is None
    assert rows[1]["warning"] == ""
