from ocsi.experiments.mot17_tracking import run_mot17_sequence

# payload = run_mot17_sequence(
#     seq_dir=r"C:\Users\User\Documents\MOT17-02-FRCNN",
#     cache_dir=r"C:\Users\User\Documents\ocsi_cache\MOT17-02-FRCNN-yolov8s-conf015",
#     output_dir=r"C:\Users\User\Documents\ocsi_outputs",
#     stages=("baseline", "memory"),
#     detection_source="public",
#     det_conf_threshold=0.30,
#     rebuild_cache=True,  # belt-and-suspenders, forces fresh build regardless
# )

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
