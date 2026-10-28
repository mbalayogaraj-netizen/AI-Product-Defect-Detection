from ultralytics import YOLO

# Load trained model
model = YOLO("model/best.pt")

# Test image
results = model.predict(
    source=r"dataset\metal_nut\test\scratch\000.png",
    conf=0.25,
    save=True,
    show=False
)

print("Detections:", len(results[0].boxes))

for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    print("Class:", cls)
    print("Confidence:", conf)