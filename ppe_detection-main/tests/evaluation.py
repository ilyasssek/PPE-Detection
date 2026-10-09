from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import yaml

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "best.pt"
DATA_YAML = BASE_DIR.parent / "dataset" / "data.yaml"

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

model = YOLO(str(MODEL_PATH))

print("Classes du modèle :", model.names)
print("Classes dataset :", data["names"])

label_counts = Counter()
for split in ["train", "valid", "test"]:
    label_dir = BASE_DIR.parent / "dataset" / split / "labels"
    if not label_dir.exists():
        continue
    for path in label_dir.glob("*.txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                cls_id = int(line.split()[0])
                label_counts[cls_id] += 1

print("\nNombre d'annotations par classe :")
for cls_id, name in data["names"].items():
    print(f"  {cls_id} ({name}) : {label_counts.get(int(cls_id), 0)}")

missing_classes = [int(cid) for cid, name in data["names"].items() if label_counts.get(int(cid), 0) == 0]
if missing_classes:
    print("\nATTENTION : classes déclarées dans data.yaml mais absentes des annotations :")
    for cid in missing_classes:
        print(f"  {cid} : {data['names'][cid]}")
    if 4 in missing_classes:
        print("\nLa classe 'vest' est déclarée mais n'apparaît dans aucune annotation de training/validation/test.")
        print("Tu dois annoter les vests avec l'ID 4 dans les fichiers .txt pour que le modèle apprenne cette classe.")

metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,
    conf=0.25,
    iou=0.5,
    plots=True
)

print(f"mAP50      : {metrics.box.map50:.6f}")
print(f"mAP50-95   : {metrics.box.map:.6f}")
print(f"Precision  : {metrics.box.mp:.6f}")
print(f"Recall     : {metrics.box.mr:.6f}")

for i, ap in enumerate(metrics.box.maps):
    print(f"{model.names[i]} : AP = {ap:.6f}")



print("\nDossier des résultats :", metrics.save_dir)