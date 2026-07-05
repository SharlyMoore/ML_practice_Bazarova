import torch
import time
import csv
import os

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

start = time.time()

results = model("images/test.jpg")

end = time.time()

detections = len(results.pandas().xyxy[0])

avg_conf = results.pandas().xyxy[0]["confidence"].mean()

file_exists = os.path.exists("results/model_comparison.csv")

with open("results/model_comparison.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(["Model", "Objects", "Average confidence", "Time (s)"])

    writer.writerow([
        "yolov5s",
        detections,
        round(avg_conf, 2),
        round(end - start, 2)
    ])

print("Готово!")