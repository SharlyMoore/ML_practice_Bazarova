from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(
    source="images/test.jpg",
    save=True,
    conf=0.25
)

print("Готово!")