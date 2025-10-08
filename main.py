from ultralytics import YOLO

# Ein vortrainiertes YOLOv8-Modell laden
model = YOLO("yolov8n.pt")

# Bildpfad
img_path = r"baustelle.jpg"   # <-- dein eigenes Bild

# Inferenz
results = model(r"C:\Users\Raphael\Documents\Automatische_Materialerfassung\Automatische-Materialerfassung\9d6bc374-f05c-497b-8927-a294b2d2c18a.webp", conf=0.25)  # conf=Schwellwert (0..1)
# Test: ein Beispielbild erkennen lassen
results[0].show()

names = model.names
for b in results[0].boxes:
    cls_id = int(b.cls[0])
    conf = float(b.conf[0])
    print(f"{names[cls_id]}: {conf:.2f}")