from ultralytics import YOLO
import cv2

model = YOLO("C:\\Users\\BeeClick\\OneDrive\\Bureau\\PFA\\ppe_detection\\model\\best2.pt")
results = model("C:\\Users\\BeeClick\\OneDrive\\Bureau\\PFA\\ppe_detection\\tests\\0308_638194157709803921.jpg", conf=0.25)

img = cv2.imread("C:\\Users\\BeeClick\\OneDrive\\Bureau\\PFA\\ppe_detection\\tests\\0308_638194157709803921.jpg")

heads = []
hats = []
vests = []

# ==============================
# 📦 DETECTIONS
# ==============================
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "head":
            heads.append((x1, y1, x2, y2))
        elif label in ["helmet", "hat"]:
            hats.append((x1, y1, x2, y2))
        elif label == "vest":
            vests.append((x1, y1, x2, y2))

# ==============================
# 🔧 بناء person box من head
# ==============================
persons = []

for (x1, y1, x2, y2) in heads:
    # نوسع box باش يمثل الجسم
    h = y2 - y1
    new_y2 = y2 + int(h * 3)  # توسعة للأسفل

    persons.append((x1, y1, x2, new_y2))

# ==============================
# CHECK INSIDE
# ==============================
def is_inside(inner, outer):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

# ==============================
# ANALYSE
# ==============================
for person in persons:
    has_hat = False
    has_vest = False

    for h in hats:
        if is_inside(h, person):
            has_hat = True

    for v in vests:
        if is_inside(v, person):
            has_vest = True

    if has_hat and has_vest:
        color = (0, 255, 0)
        text = "SAFE"
    else:
        color = (0, 0, 255)
        text = "NOT SAFE"

    x1, y1, x2, y2 = person

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("result.jpg", img)