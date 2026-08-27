"""Tests for the OCSI command-line wrapper."""
import csv
import json

from ocsi import cli


def test_mot17_q1_cli_writes_flat_and_summary_outputs(tmp_path, monkeypatch):
    data_root = tmp_path / "MOT17"
    cache_root = tmp_path / "cache"
    out_dir = tmp_path / "out"
    (data_root / "MOT17-02-FRCNN").mkdir(parents=True)
    cache_root.mkdir()

    def fake_run_mot17_dataset(**kwargs):
        assert kwargs["stages"] == ("baseline", "memory", "adaptive_memory")
        assert kwargs["detection_source"] == "public"
        return {
            "runs": [
                {
                    "sequence": "MOT17-02-FRCNN",
                    "seed": 1,
                    "recommended_reactivation_app_gate": 0.88,
                    "embedding_diagnostics": {"separation_margin": 0.15},
                    "results": [
                        {
                            "stage": "baseline",
                            "result_path": "baseline.txt",
                            "metrics": {
                                "idf1": 0.5,
                                "mota": 0.4,
                                "idsw": 10,
                                "precision": 0.8,
                                "recall": 0.7,
                            },
                        },
                        {
                            "stage": "memory",
                            "result_path": "memory.txt",
                            "metrics": {
                                "idf1": 0.55,
                                "mota": 0.42,
                                "idsw": 8,
                                "precision": 0.81,
                                "recall": 0.71,
                            },
                        },
                        {
                            "stage": "adaptive_memory",
                            "result_path": "adaptive.txt",
                            "metrics": {
                                "idf1": 0.58,
                                "mota": 0.43,
                                "idsw": 7,
                                "precision": 0.82,
                                "recall": 0.72,
                            },
                        },
                    ],
                }
            ]
        }

    monkeypatch.setattr(cli, "run_mot17_dataset", fake_run_mot17_dataset)

    assert cli.main(
        [
            "mot17-q1",
            "--data-root",
            str(data_root),
            "--cache-root",
            str(cache_root),
            "--out",
            str(out_dir),
            "--sequences",
            "MOT17-02-FRCNN",
            "--seeds",
            "1",
        ]
    ) == 0

    with (out_dir / "per_sequence_results.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["Stage"] for row in rows] == ["baseline", "memory", "adaptive_memory"]
    assert rows[2]["RecommendedGate"] == "0.88"

    assert (out_dir / "paper_summary_bootstrap_ci.csv").exists()
    assert (out_dir / "paper_paired_statistics.csv").exists()
    payload = json.loads((out_dir / "q1_payload.json").read_text(encoding="utf-8"))
    assert payload["runs"][0]["sequence"] == "MOT17-02-FRCNN"
