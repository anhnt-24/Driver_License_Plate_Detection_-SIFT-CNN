import csv
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter_area = (xB - xA) * (yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

csv_path = 'bounding_boxes.csv'
label_dir = 'dataset/test/labels'
image_dir = 'dataset/test/images'

results = []
iou_threshold = 0.5

with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = row['filename']
        try:
            csv_box = list(map(int, [row['min_x'], row['min_y'], row['max_x'], row['max_y']]))
        except ValueError:
            print(f"Dữ liệu lỗi trong CSV: {filename}")
            continue

        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(label_path) or not os.path.exists(image_path):
            print(f"Thiếu ảnh hoặc label: {filename}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"⚠Không có label trong file: {filename}")
            continue

        max_iou = 0.0
        best_label_box = None
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 9 or len(parts) % 2 == 0:
                continue

            coords = parts[1:]
            points = np.array([[coords[i] * width, coords[i + 1] * height] for i in range(0, len(coords), 2)])
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            label_box = [x_min, y_min, x_max, y_max]

            iou = compute_iou(csv_box, label_box)
            if iou > max_iou:
                max_iou = iou
                best_label_box = label_box

        results.append((filename, max_iou))


        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle(csv_box, outline="red", width=3)
            draw.rectangle(best_label_box, outline="lime", width=3)
            plt.figure(figsize=(8, 6))
            plt.title(f"{filename} (IoU = {max_iou:.2f}) - Đỏ: CSV | Xanh: Label")
            plt.imshow(img)
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {str(e)}")
matched = [iou for _, iou in results if iou >= iou_threshold]
avg_iou = sum(iou for _, iou in results) / len(results) if results else 0.0
match_ratio = len(matched) / len(results) if results else 0.0

print("\n Kết quả từng ảnh:")
for filename, iou in results:
    print(f"{filename}: IoU cao nhất = {iou:.4f}")

print("\n Thống kê:")
print(f"Tổng số ảnh: {len(results)}")
print(f"Số ảnh khớp (IoU ≥ {iou_threshold}): {len(matched)}")
print(f"Tỷ lệ khớp đúng: {match_ratio * 100:.2f}%")
print(f"IoU trung bình: {avg_iou:.4f}")
