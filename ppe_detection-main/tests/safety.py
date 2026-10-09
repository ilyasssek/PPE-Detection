from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os
import csv
from collections import defaultdict

MODEL_PATH = r"C:\Users\BeeClick\OneDrive\Bureau\PFA\ppe_detection\model\best.pt"
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

# ========== Configuration des logs ==========
LOG_DIR = "logs"
SCREENSHOT_DIR = "screenshots"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"ppe_log_{datetime.now().strftime('%Y%m%d')}.csv")

if not os.path.exists(log_file):
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'safe_count', 'not_safe_count', 'total_persons', 'screenshot'])


person_tracking = {}
PERSON_ID = 0  # Compteur pour assigner des IDs uniques

# Configuration du filtrage
FRAMES_REQUIRED = 5  # Nombre de frames consécutives nécessaires pour changer le statut
MAX_PERSON_ID = 100  # Garder les IDs des 100 dernières personnes

# Pour le suivi simple par position (sans DeepSORT)
def get_person_id(person_box, previous_persons, iou_threshold=0.3):
    """
    Associe une personne à un ID existant basé sur l'IoU avec la frame précédente
    """
    best_iou = 0
    best_id = None
    
    for pid, prev_box in previous_persons.items():
        iou = box_iou(person_box, prev_box)
        if iou > best_iou:
            best_iou = iou
            best_id = pid
    
    if best_iou > iou_threshold:
        return best_id
    else:
        return None

def save_screenshot(frame, status, timestamp):
    filename = f"screenshot_{timestamp.strftime('%Y%m%d_%H%M%S')}_{status}.jpg"
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filename

def write_log(timestamp, safe, not_safe, total, screenshot_name):
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, safe, not_safe, total, screenshot_name])

def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0

def is_hat_for_person(hat, person):
    hat_cx, hat_cy = box_center(hat)
    person_width = person[2] - person[0]
    person_height = person[3] - person[1]

    if not (person[0] - person_width * 0.1 <= hat_cx <= person[2] + person_width * 0.1):
        return False
    if hat_cy > person[1] + person_height * 0.35:
        return False
    return box_iou(hat, person) >= 0.01 or hat_cy < person[1] + person_height * 0.25

def is_vest_for_person(vest, person):
    vest_cx, vest_cy = box_center(vest)
    person_width = person[2] - person[0]
    person_height = person[3] - person[1]

    if not (person[0] - person_width * 0.1 <= vest_cx <= person[2] + person_width * 0.1):
        return False
    if not (person[1] + person_height * 0.25 <= vest_cy <= person[3] + person_height * 0.1):
        return False
    return box_iou(vest, person) >= 0.02 or (vest[1] < person[3] and vest[3] > person[1] + person_height * 0.25)

# Variables pour le suivi inter-frame
previous_persons = {}  # {person_id: box}
current_persons_info = {}  
last_screenshot_time = None
SCREENSHOT_COOLDOWN = 3

# ========== BOUCLE PRINCIPALE ==========
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    frame_count += 1
    
    safe_count = 0
    not_safe_count = 0

    results = model(frame, conf=0.3)[0]

    persons = []
    hats_on = []
    vests_on = []

    for box in results.boxes:
        xyxy = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        name = model.names[cls]
        
        if name == "person":
            persons.append(xyxy)
        elif name == "hat":
            hats_on.append(xyxy)
        elif name == "vest":
            vests_on.append(xyxy)

    # ========== NOUVEAU : Traitement avec filtrage temporel ==========
    current_persons = {}  # {person_id: box} pour cette frame
    
    # 1. Détecter le statut BRUT pour chaque personne (sans filtre)
    raw_statuses = []
    
    for idx, person in enumerate(persons):
        has_hat = False
        has_vest = False

        for hat in hats_on:
            if is_hat_for_person(hat, person):
                has_hat = True
                break

        for vest in vests_on:
            if is_vest_for_person(vest, person):
                has_vest = True
                break
        
        raw_safe = has_hat and has_vest
        raw_statuses.append((person, raw_safe))
    
    # 2. Associer chaque personne à un ID (basé sur l'IoU avec frame précédente)
    # Réinitialiser current_persons_info pour cette frame
    new_person_info = {}
    
    for person, raw_safe in raw_statuses:
        # Trouver un ID existant ou en créer un nouveau
        matched_id = get_person_id(person, previous_persons)
        
        if matched_id is None:
            # Nouvelle personne : créer un nouvel ID
            person_id = PERSON_ID
            PERSON_ID += 1
            # Initialiser le suivi
            person_tracking[person_id] = {
                'status': 'SAFE' if raw_safe else 'NOT_SAFE',
                'consecutive_frames': 1
            }
        else:
            person_id = matched_id
            # Mettre à jour le compteur de frames consécutives
            current_status = 'SAFE' if raw_safe else 'NOT_SAFE'
            previous_status = person_tracking[person_id]['status']
            
            if current_status == previous_status:
                # Même statut : incrémenter
                person_tracking[person_id]['consecutive_frames'] += 1
            else:
                # Changement de statut : réinitialiser
                person_tracking[person_id]['consecutive_frames'] = 1
                person_tracking[person_id]['status'] = current_status
        
        # Déterminer le statut AFFICHÉ (après filtrage)
        if person_tracking[person_id]['consecutive_frames'] >= FRAMES_REQUIRED:
            display_status = person_tracking[person_id]['status']
        else:
            # Pendant la période de stabilisation, garder l'ancien statut ou 'SAFE' par défaut
            if matched_id is None:
                display_status = 'SAFE'  # Nouvelle personne : safe par défaut
            else:
                # Garder l'ancien statut
                display_status = previous_status if 'previous_status' in locals() else 'SAFE'
        
        # Stocker les infos pour cette frame
        new_person_info[person_id] = {
            'box': person,
            'raw_status': 'SAFE' if raw_safe else 'NOT_SAFE',
            'display_status': display_status,
            'consecutive_frames': person_tracking[person_id]['consecutive_frames']
        }
        
        # Compter pour l'affichage
        if display_status == 'SAFE':
            safe_count += 1
        else:
            not_safe_count += 1
    
    # Nettoyer les anciens IDs (limiter la mémoire)
    if PERSON_ID > MAX_PERSON_ID:
        # Garder seulement les IDs récents
        recent_ids = list(new_person_info.keys())
        person_tracking = {pid: person_tracking[pid] for pid in recent_ids if pid in person_tracking}
        PERSON_ID = max(recent_ids) + 1 if recent_ids else 0
    
    # Mettre à jour previous_persons pour la prochaine frame
    previous_persons = {pid: info['box'] for pid, info in new_person_info.items()}
    
    # ========== AFFICHAGE ==========
    for person_id, info in new_person_info.items():
        x1, y1, x2, y2 = map(int, info['box'])
        
        if info['display_status'] == 'SAFE':
            label = "SAFE"
            color = (0, 255, 0)
        else:
            label = "NOT SAFE"
            color = (0, 0, 255)
        
        # Afficher aussi le compteur de frames pour debug
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        
        # Petit texte pour voir le filtrage en action (debug)
        cv2.putText(frame, f"conf: {info['consecutive_frames']}/{FRAMES_REQUIRED}",
                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # ========== RÉSUMÉ ET CAPTURES ==========
    total_persons = safe_count + not_safe_count
    
    # Afficher le résumé
    cv2.putText(frame, f"SAFE: {safe_count} | NOT SAFE: {not_safe_count} | TOTAL: {total_persons}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Info filtrage
    cv2.putText(frame, f"FILTRAGE: {FRAMES_REQUIRED} frames consecutives requises",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Capture automatique (seulement après confirmation du filtrage)
    current_time = datetime.now()
    if not_safe_count > 0:
        if last_screenshot_time is None or (current_time - last_screenshot_time).seconds >= SCREENSHOT_COOLDOWN:
            screenshot_name = save_screenshot(frame, "ALERT", current_time)
            print(f"📸 ALERTE - Capture sauvegardée : {screenshot_name}")
            last_screenshot_time = current_time
            write_log(current_time, safe_count, not_safe_count, total_persons, screenshot_name)
    
    # Afficher les détections pour debug
    for hat in hats_on:
        x1, y1, x2, y2 = map(int, hat)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "HAT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    for vest in vests_on:
        x1, y1, x2, y2 = map(int, vest)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, "VEST", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    cv2.imshow("Detection PPE - Filtrage temporel", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📊 Session terminée - Log sauvegardé dans : {log_file}")
print(f"📸 Captures sauvegardées dans : {SCREENSHOT_DIR}")