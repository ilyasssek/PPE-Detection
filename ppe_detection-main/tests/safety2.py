import cv2
from ultralytics import YOLO

# Load model for the 3 classes: Helmet, Vest, Head
model = YOLO(r"C:\Users\BeeClick\OneDrive\Bureau\PFA\ppe_detection\model\best3.pt")
names = model.names
print("Classes:", names)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def is_helmet_on_head(head, helmet):
    if box_iou(head, helmet) > 0.02:
        return True

    hx1, hy1, hx2, hy2 = head
    hw = hx2 - hx1
    cx = (helmet[0] + helmet[2]) / 2
    cy = (helmet[1] + helmet[3]) / 2

    if cx < hx1 - hw * 0.2 or cx > hx2 + hw * 0.2:
        return False
    return cy < hy2 + (hy2 - hy1) * 0.25


def is_vest_for_head(head, vest):
    if box_iou(head, vest) > 0.02:
        return True

    hx1, hy1, hx2, hy2 = head
    hw = hx2 - hx1
    cx = (vest[0] + vest[2]) / 2
    if cx < hx1 - hw * 0.25 or cx > hx2 + hw * 0.25:
        return False
    return vest[1] > hy2 - (hy2 - hy1) * 0.2


def box_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def are_near(a, b, max_ratio=0.7):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw = ax2 - ax1
    bw = bx2 - bx1
    return abs((ax1 + ax2) / 2 - (bx1 + bx2) / 2) < max(aw, bw) * max_ratio


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    results = model(frame, conf=0.3)
    heads = []
    helmets = []
    vests = []

    if len(results) > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = names[cls].lower()

            if label == "head":
                heads.append((x1, y1, x2, y2))
            elif label == "helmet":
                helmets.append((x1, y1, x2, y2))
            elif label == "vest":
                vests.append((x1, y1, x2, y2))

    safe_count = 0
    not_safe_count = 0
    used_helmet_idxs = set()
    used_vest_idxs = set()

    for head in heads:
        has_helmet = False
        has_vest = False

        for hi, h in enumerate(helmets):
            if is_helmet_on_head(head, h):
                has_helmet = True
                used_helmet_idxs.add(hi)
                break

        for vi, v in enumerate(vests):
            if is_vest_for_head(head, v):
                has_vest = True
                used_vest_idxs.add(vi)
                break

        if has_helmet and has_vest:
            color = (0, 255, 0)
            label = "SAFE"
            safe_count += 1
        else:
            color = (0, 0, 255)
            label = "NOT SAFE"
            not_safe_count += 1

        cv2.rectangle(frame, (head[0], head[1]), (head[2], head[3]), color, 3)
        cv2.putText(frame, label, (head[0], head[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    pseudo_safe_boxes = []
    for hi, h in enumerate(helmets):
        if hi in used_helmet_idxs:
            continue
        for vi, v in enumerate(vests):
            if vi in used_vest_idxs:
                continue
            if are_near(h, v) and v[1] > h[1]:
                pseudo_safe_boxes.append(box_union(h, v))
                used_helmet_idxs.add(hi)
                used_vest_idxs.add(vi)
                break

    for box in pseudo_safe_boxes:
        safe_count += 1
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.putText(frame, "SAFE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mark unmatched single PPE as NOT SAFE
    for hi, h in enumerate(helmets):
        if hi in used_helmet_idxs:
            continue
        not_safe_count += 1
        cv2.rectangle(frame, (h[0], h[1]), (h[2], h[3]), (0, 0, 255), 2)
        cv2.putText(frame, "NOT SAFE", (h[0], h[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for vi, v in enumerate(vests):
        if vi in used_vest_idxs:
            continue
        not_safe_count += 1
        cv2.rectangle(frame, (v[0], v[1]), (v[2], v[3]), (0, 0, 255), 2)
        cv2.putText(frame, "NOT SAFE", (v[0], v[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"SAFE: {safe_count} NOT SAFE: {not_safe_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()