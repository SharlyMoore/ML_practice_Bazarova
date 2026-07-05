from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("images/test.jpg")

result = results[0]

for box in result.boxes:
    class_id = int(box.cls)
    confidence = float(box.conf)

    print(
        f"Объект: {model.names[class_id]}, "
        f"уверенность: {confidence:.2f}"
    )