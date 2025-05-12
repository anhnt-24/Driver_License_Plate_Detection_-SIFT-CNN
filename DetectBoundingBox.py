import csv
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from Sift_v2 import *

def draw_keypoints_with_match(image, keypoints, matches):
    img_with_keypoints = image.copy()
    pts2 = np.float32([ (kp['pt'][0], kp['pt'][1]) for _, j in matches for kp in [keypoints[j]] ])

    for x, y in pts2:
        cv2.circle(img_with_keypoints, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=2)

    return img_with_keypoints

def process_descriptor_file(desc_file_path, desc_query):
    with open(desc_file_path, 'rb') as f:
        data = pickle.load(f)
        desc = data['descriptor']

        matches = match_keypoints_bf(desc, desc_query)
        match_count = len(matches)

        return {
            'file_path': desc_file_path,
            'desc': desc,
            'matches': matches,
            'match_count': match_count
        }


def get_bounding_box(file_path):
    img_query = cv2.imread(file_path)

    sift=SiftHandler("Query",img_query)
    sift.exec()
    desc_query=sift.descriptors
    keypoints=sift.keypoints
    desc_folder = "desc"
    desc_files = [os.path.join(desc_folder, f) for f in os.listdir(desc_folder) if f.endswith(".pkl")]

    best_match_data = None
    best_match_count = -1

    max_workers = min(os.cpu_count(), 8)
    matches_keypoints=[]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_descriptor_file, file_path, desc_query): file_path
            for file_path in desc_files
        }

        for i, future in enumerate(as_completed(future_to_file), 1):
            try:
                result = future.result()
                match_count = result['match_count']
                file_name = os.path.basename(result['file_path'])

                print(f"Progress: {i}/{len(desc_files)} - {file_name}: {match_count} matches")

                if result['matches']:  # nếu có match
                    matches_keypoints += result['matches']
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_match_data = result

            except Exception as e:
                print(f"Error processing {future_to_file[future]}: {e}")

    if best_match_data:
        file_path = best_match_data['file_path']
        file_name = os.path.basename(file_path)
        print(f"\nBest match: {file_name} with {best_match_count} matches")

        best_img_name = file_name.replace("_desc_oriented.pkl", "") + ".jpg"
        best_img_path = os.path.join("dataset/cropped", best_img_name)

        try:
            img_best = cv2.imread(best_img_path, cv2.IMREAD_GRAYSCALE)

            if img_best is None:
                print(f"Could not load best match image from {best_img_path}")
                return

            result,cropped_region,bbox = draw_bounding_box_on_img2(
                sift.base_image,
                keypoints,
                best_match_data['matches']
            )

            # cv2.imshow("keypoints2", draw_keypoints_with_match(sift.base_image, keypoints, matches_keypoints))
            # cv2.imshow("cropped_region", result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return result, cropped_region,bbox

        except Exception as e:
            print(f"Error displaying results: {e}")
    else:
        print("No matching images found.")


if __name__ == "__main__":
    input_dir = "dataset/test/images"
    csv_path = "bounding_boxes.csv"

    existing_files = set()
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)
            for row in reader:
                if row:
                    existing_files.add(row[0])
    cnt=0
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        if os.stat(csv_path).st_size == 0:
            writer.writerow(['filename', 'min_x', 'min_y', 'max_x', 'max_y'])

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                cnt+=1
                if filename in existing_files:
                    print(f"Bỏ qua {filename}, đã có trong CSV.")

                    continue
                print(cnt)
                full_path = os.path.join(input_dir, filename)
                output = get_bounding_box(full_path)
                if output is not None:
                    result, cropped_region, bbox = output
                    if cropped_region is not None:
                        writer.writerow([filename, *bbox])
