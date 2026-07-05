from ultralytics import YOLO
import csv

model = YOLO("yolov8n.pt")

results = model("images/test.jpg")

result = results[0]

rows = []

for box in result.boxes:
    class_id = int(box.cls)
    confidence = float(box.conf)

    rows.append([
        model.names[class_id],
        round(confidence, 2)
    ])

with open("results/results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(["Object", "Confidence"])

    writer.writerows(rows)

print("Файл results.csv сохранен!")