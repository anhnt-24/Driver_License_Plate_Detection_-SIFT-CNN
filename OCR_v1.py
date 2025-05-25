import numpy as np
import cv2
import keras
from matplotlib import pyplot as plt

from rotate_and_recrop_lp import auto_rotate_and_crop_lp
load_model=keras.models.load_model
import imutils
from skimage import measure
from sklearn.cluster import KMeans



def convert2Square(image):
    h, w = image.shape
    if w > h:
        diff = w - h
        top, bottom = diff // 2, diff - (diff // 2)
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        diff = h - w
        left, right = diff // 2, diff - (diff // 2)
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
    return image


def find_contours_unified(binary_img):
    height, width = binary_img.shape
    char_data = []
    labels = measure.label(binary_img, connectivity=2, background=0)
    regions = measure.regionprops(labels)

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        h = max_row - min_row
        w = max_col - min_col
        y = min_row
        x = min_col

        aspectRatio = w / float(h) if h > 0 else 0
        area = region.area
        solidity = float(area) / (w * h) if w * h > 0 else 0
        heightRatio = h / float(height)

        if (w >= 5 and h >= 8 and
                w < width / 1.5 and h < height / 1.2 and
                0.1 < aspectRatio < 1.5 and
                solidity > 0.1 and
                0.08 < heightRatio < 0.98):
            char_data.append({
                'image': process_character(binary_img, x, y, w, h),
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center_y': y + h / 2
            })
    inverted_binary = cv2.bitwise_not(binary_img.copy())
    processed = cv2.medianBlur(inverted_binary, 3)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        aspectRatio = w / float(h) if h > 0 else 0
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / float(w * h) if w * h > 0 else 0
        heightRatio = h / float(height)

        if (w >= 5 and h >= 8 and
                w < width / 1.5 and h < height / 1.2 and
                0.1 < aspectRatio < 1.5 and
                solidity > 0.1 and
                0.08 < heightRatio < 0.98):

            is_duplicate = False
            for existing in char_data:
                overlap_x = max(0, min(existing['x'] + existing['w'], x + w) - max(existing['x'], x))
                overlap_y = max(0, min(existing['y'] + existing['h'], y + h) - max(existing['y'], y))
                overlap_area = overlap_x * overlap_y
                min_area = min(existing['w'] * existing['h'], w * h)

                if overlap_area > 0.5 * min_area:
                    is_duplicate = True
                    break

            if not is_duplicate:
                char_data.append({
                    'image': process_character(binary_img, x, y, w, h),
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'center_y': y + h / 2
                })

    if len(char_data) < 4:
        thresh = binary_img.copy()
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspectRatio = w / float(h) if h > 0 else 0
            solidity = cv2.contourArea(contour) / float(w * h) if w * h > 0 else 0
            heightRatio = h / float(height)
            if (w >= 3 and h >= 8 and
                    w <= width / 1.2 and h <= height / 1.2 and
                    0.05 < aspectRatio < 1.8 and
                    solidity > 0.08 and
                    0.08 < heightRatio < 0.98):

                is_duplicate = False
                for existing in char_data:
                    overlap_x = max(0, min(existing['x'] + existing['w'], x + w) - max(existing['x'], x))
                    overlap_y = max(0, min(existing['y'] + existing['h'], y + h) - max(existing['y'], y))
                    overlap_area = overlap_x * overlap_y
                    min_area = min(existing['w'] * existing['h'], w * h)

                    if overlap_area > 0.3 * min_area:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    char_data.append({
                        'image': process_character(binary_img, x, y, w, h),
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'center_y': y + h / 2
                    })
    return char_data


def process_character(binary_img, x, y, w, h):
    char = binary_img[y:y + h, x:x + w]
    char_copy = np.zeros((44, 24))
    char = cv2.resize(char, (20, 40))
    char_copy[2:42, 2:22] = char
    char_copy[0:2, :] = 0
    char_copy[:, 0:2] = 0
    char_copy[42:44, :] = 0
    char_copy[:, 22:24] = 0

    return char_copy


def determine_plate_type_and_order(char_data, plate_height):
    if not char_data:
        return [], 'unknown'
    y_centers = np.array([char['center_y'] for char in char_data])

    if len(y_centers) > 3:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(y_centers.reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_

        vertical_distance = abs(centers[0] - centers[1])

        if vertical_distance > plate_height * 0.2:
            upper_chars = [char for char, label in zip(char_data, labels) if
                           label == (0 if centers[0] < centers[1] else 1)]
            lower_chars = [char for char, label in zip(char_data, labels) if
                           label == (1 if centers[0] < centers[1] else 0)]
            upper_chars.sort(key=lambda c: c['x'])
            lower_chars.sort(key=lambda c: c['x'])
            sorted_chars = upper_chars + lower_chars
            return [char['image'] for char in sorted_chars], 'two-line'

    char_data.sort(key=lambda c: c['x'])
    return [char['image'] for char in char_data], 'one-line'


def preprocess_plate_image1(plate_image):
    img_gray_lp = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))
    img_binary_lp = cv2.bitwise_not(img_binary_lp)

    return img_binary_lp


def preprocess_plate_image2(plate_image):
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    v = cv2.GaussianBlur(v, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel_close = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            cv2.drawContours(thresh, [cnt], -1, 0, -1)

    return thresh


def preprocess_plate_image3(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, width=400)
    thresh = cv2.medianBlur(thresh, 5)
    thresh = cv2.bitwise_not(thresh)

    return thresh


def lp_char_recog(plate_image, show_image=False):
    plate_height, plate_width = plate_image.shape[:2]
    aspect_ratio = plate_width / plate_height

    if (aspect_ratio > 2.0):
        plate_image = cv2.resize(plate_image, (350, 100))
    else:
        plate_image = cv2.resize(plate_image, (240, 200))

    angle, binary_img, rotated_lp = auto_rotate_and_crop_lp(plate_image)

    preprocessing_methods = [
        {"name": "Method 3", "function": preprocess_plate_image3},
        {"name": "Method 1", "function": preprocess_plate_image1},
        {"name": "Method 2", "function": preprocess_plate_image2},
    ]

    all_results = []

    for method in preprocessing_methods:
        img_binary = method["function"](rotated_lp)

        char_data = find_contours_unified(img_binary)

        all_results.append({
            "method_name": method["name"],
            "char_count": len(char_data),
            "char_data": char_data,
            "img_binary": img_binary
        })

    all_good_results = all([result["char_count"] > 5 for result in all_results])

    if all_good_results:
        for result in all_results:
            if result["method_name"] == "Method 3":
                best_char_data = result["char_data"]
                best_img_binary = result["img_binary"]
                best_method_name = "Method 3 (prioritized)"
                best_char_count = result["char_count"]
                break
    else:
        best_char_count = 0
        best_char_data = []
        best_img_binary = None
        best_method_name = ""

        for result in all_results:
            if result["char_count"] > best_char_count and result["char_count"] <= 10:
                best_char_count = result["char_count"]
                best_char_data = result["char_data"]
                best_img_binary = result["img_binary"]
                best_method_name = result["method_name"]

    print(f"Using {best_method_name} with {best_char_count} characters detected")

    if len(best_char_data) == 0:
        print("No characters detected with standard methods. Trying special handling for one-line plates...")

        if aspect_ratio > 2.0:
            one_line_plate = cv2.resize(rotated_lp, (333, 75))

            gray_plate = cv2.cvtColor(one_line_plate, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_plate, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


            height, width = binary.shape
            char_data = []
            vis_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                if area > 30 and w > 3 and h > 8 and h < height * 0.95:
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    char = binary[y:y + h, x:x + w]
                    char_sq = convert2Square(char)
                    char_resize = cv2.resize(char_sq, (28, 28))

                    char_data.append({
                        'image': char_resize,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'center_y': y + h / 2
                    })

            if len(char_data) > 0:
                best_char_data = char_data
                best_img_binary = binary
            else:
                return "No characters detected in one-line plate"
        else:
            return "No characters detected"

    segmented_chars, plate_type = determine_plate_type_and_order(best_char_data, best_img_binary.shape[0])
    print(f"Detected plate type: {plate_type}")
    # plt.figure(figsize=(12, 3))
    # for i, char_img in enumerate(segmented_chars):
    #     plt.subplot(1, len(segmented_chars), i + 1)
    #     plt.imshow(char_img, cmap='gray')
    #     plt.title(f"Char {i + 1}")
    #     plt.axis('off')
    # plt.suptitle(f"Segmented Characters (Plate Type: {plate_type})")
    # plt.show()
    # Load model
    model_path = 'models/weight.keras'
    model = load_model(model_path)

    char_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K',
                 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'R', 14: 'S', 15: 'T', 16: 'U',
                 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
                 25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

    output = []
    for char_img in segmented_chars:
        img = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.reshape(28, 28, 1)
        img_input = img.reshape(1, 28, 28, 1)
        pred = model.predict(img_input)
        char_idx = np.argmax(pred, axis=-1)[0]
        predicted_char = char_dict[char_idx]

        if predicted_char != "Background":
            output.append(predicted_char)

    plate_number = ''.join(output)
    return plate_number


if __name__ == "__main__":

    def test_recognition(path):
        plate_image = cv2.imread(path)
        if plate_image is None:
            print(f"Failed to load image: {path}")
            return
        result = lp_char_recog(plate_image)
        print(f"Recognized license plate: {result}")


    test_recognition("dataset/cropped/greenpack_0169_0.jpg")
