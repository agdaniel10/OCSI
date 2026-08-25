from ocsi.experiments.mot17_tracking import run_mot17_sequence

payload = run_mot17_sequence(
    seq_dir=r"C:\Users\User\Documents\MOT17-02-FRCNN",
    cache_dir=r"C:\Users\User\Documents\ocsi_cache\MOT17-02-FRCNN-yolov8s-conf015",
    output_dir=r"C:\Users\User\Documents\ocsi_outputs",
    stages=("baseline", "memory", "feedback"),
    detection_source="public",
    det_conf_threshold=0.30,
    rebuild_cache=True,  # belt-and-suspenders, forces fresh build regardless
)

for r in payload["results"]:
    print(r["stage"], r["summary"])

diag = payload["embedding_diagnostics"]
print("\nembedding diagnostics")
print("  assigned embeddings:", diag["assigned_embeddings"], "/", diag["total_embeddings"])
print("  identity prototypes:", diag["num_identity_prototypes"])
print("  embedding dim:", diag["embedding_dim"])
print("  mean norm:", f"{diag['mean_norm']:.3f}", "+/-", f"{diag['std_norm']:.3f}")
print("  same-id cosine:", diag["same_id_proto_cosine"])
print("  diff-id cosine:", diag["different_id_proto_cosine"])
print("  separation margin:", diag["separation_margin"])
