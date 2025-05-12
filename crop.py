import os
import cv2
import numpy as np

dataset_path = './dataset'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')
cropped_path = os.path.join(dataset_path, 'cropped')

os.makedirs(cropped_path, exist_ok=True)

for label_file in os.listdir(labels_path):
    if not label_file.endswith('.txt'):
        continue

    label_path = os.path.join(labels_path, label_file)
    image_name = os.path.splitext(label_file)[0]

    image_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        temp_path = os.path.join(images_path, image_name + ext)
        if os.path.exists(temp_path):
            image_path = temp_path
            break

    if image_path is None:
        print(f"Không tìm thấy ảnh cho {label_file}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        continue

    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = list(map(float, line.strip().split()))
        if len(parts) < 9 or len(parts) % 2 == 0:
            print(f"Dòng {i+1} trong {label_file} không hợp lệ.")
            continue

        coords = parts[1:]
        pts = np.array([[coords[j] * w, coords[j + 1] * h] for j in range(0, len(coords), 2)], dtype=np.float32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

        masked = cv2.bitwise_and(image, image, mask=mask)

        x, y, bw, bh = cv2.boundingRect(pts)

        if bw == 0 or bh == 0:
            print(f"Bounding box không hợp lệ cho {image_name}_{i}. Không cắt.")
            continue

        cropped = masked[y:y+bh, x:x+bw]

        if cropped.size == 0:
            print(f"Ảnh cắt rỗng cho {image_name}_{i}.")
            continue

        crop_filename = f"{image_name}_{i}.jpg"
        cv2.imwrite(os.path.join(cropped_path, crop_filename), cropped)

    print(f"Đã xử lý: {image_name}")

