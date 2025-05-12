import pickle
from Sift_v2 import *

os.makedirs('desc', exist_ok=True)
os.makedirs('info', exist_ok=True)

info_path = os.path.join('info', 'image_info.csv')
info_file = open(info_path, 'w', encoding='utf-8')
info_file.write('filename,width,height,num_keypoints,elapsed_time\n')  # header

image_folder = 'dataset/cropped'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

cnt = 0
for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path)

    start_time = time.time()

    sift = SiftHandler(img_path, img)
    sift.exec()

    elapsed_time = time.time() - start_time

    height, width = img.shape[:2]
    num_keypoints = len(sift.keypoints) if sift.keypoints is not None else 0

    desc_filename = os.path.join('desc', f"{os.path.splitext(image_file)[0]}_desc_oriented.pkl")
    with open(desc_filename, 'wb') as f:
        pickle.dump({'descriptor': sift.descriptors}, f)

    info_file.write(f"{image_file},{width},{height},{num_keypoints},{elapsed_time:.4f}\n")

    print(f"✓ Đã lưu: {desc_filename}")
    cnt += 1
    print(f"   Tiến độ: {cnt}/{len(image_files)} ảnh | Thời gian: {elapsed_time:.2f}s")

info_file.close()
print(f"✔️ Đã lưu thông tin vào: {info_path}")
