from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = "C:\\Users\\BeeClick\\OneDrive\\Bureau\\PFA\\ppe_detection\\model\\best.pt"
VALID_IMAGES = "C:\\Users\\BeeClick\\OneDrive\\Bureau\\PFA\\ppe_detection\\dataset\\valid\\images"

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(VALID_IMAGES),
    conf=0.25,
    save=True,
    imgsz=640
)

print("Résultats sauvegardés dans runs/detect/predict")