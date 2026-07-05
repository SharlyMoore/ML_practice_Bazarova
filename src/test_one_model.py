import csv
import os
import time
from ultralytics import YOLO

# MODEL_NAME = "yolov8n.pt"
# MODEL_NAME = "yolov8m.pt"
# MODEL_NAME = "yolov8s.pt"
# MODEL_NAME = "yolov8l.pt"
MODEL_NAME = "yolov8x.pt"

model = YOLO(MODEL_NAME)

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

file_exists = os.path.exists("results/model_comparison.csv")

with open("results/model_comparison.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "Model",
            "Objects",
            "Average confidence",
            "Time (s)"
        ])

    writer.writerow([
        MODEL_NAME,
        detections,
        round(avg_conf, 2),
        round(end - start, 2)
    ])

print("Готово!")