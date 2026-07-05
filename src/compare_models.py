import csv
import time
from ultralytics import YOLO

models = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt"
]

rows = []

for model_name in models:
    print(f"Тестируется {model_name}...")

    model = YOLO(model_name)

    start = time.time()
    results = model.predict(
        source="images/test.jpg",
        verbose=False
    )
    end = time.time()

    result = results[0]

    detections = len(result.boxes)

    avg_conf = (
        sum(float(box.conf) for box in result.boxes) / detections
        if detections else 0
    )

    rows.append([
        model_name,
        detections,
        round(avg_conf, 2),
        round(end - start, 2)
    ])

with open("results/model_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model",
        "Objects",
        "Average confidence",
        "Time (s)"
    ])
    writer.writerows(rows)

print("\nГотово!")
print("Файл сохранен: results/model_comparison.csv")