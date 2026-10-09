from ultralytics import YOLO
import cv2
from pathlib import Path

# =========================
# Chargement du modèle
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = r"C:\Users\BeeClick\OneDrive\Bureau\PFA\ppe_detection\model\best.pt"
model = YOLO(str(MODEL_PATH))
print("Classes du modèle :", model.names)

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : webcam introuvable.")
    exit()

# Classes à afficher
target_classes = {"vest", "novest", "hat", "nohat", "person"}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lecture webcam.")
        break

    frame = cv2.flip(frame, 1)

    # Prédiction
    results = model(frame, conf=0.3)[0]

    detected_classes = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        if class_name in target_classes:
            detected_classes.append(class_name)

            # Couleurs par classe
            if class_name == "person":
                color = (255, 255, 255)
            elif class_name in ["hat", "vest"]:
                color = (0, 255, 0)
            else:  # nohat, novest
                color = (0, 0, 255)

            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Résumé en haut
    summary = " | ".join(sorted(set(detected_classes))) if detected_classes else "Aucune detection"
    cv2.putText(
        frame,
        f"Detecte: {summary}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("Test detection classes", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()