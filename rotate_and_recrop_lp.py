import cv2
import numpy as np
import math


def preprocess(img):
    imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgGrayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return imgGrayscale, imgThresh


def Hough_transform(img, nol=6):
    linesP = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None and len(linesP) > nol:
        linesP = linesP[:nol]
    return linesP


def rotation_angle(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 45:
            angles.append(angle)
    if angles:
        return np.median(angles)
    return 0


def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated


def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def auto_crop_license_plate(img, binary_img):
    contours = find_contours(binary_img)

    if not contours:
        return img

    largest_contour = contours[0]
    x, y, w, h = cv2.boundingRect(largest_contour)

    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)

    img_h, img_w = img.shape[:2]
    x_start = max(0, x - margin_x)
    y_start = max(0, y - margin_y)
    x_end = min(img_w, x + w + margin_x)
    y_end = min(img_h, y + h + margin_y)

    return img[y_start:y_end, x_start:x_end]


def auto_rotate_and_crop_lp(source_img):
    if source_img is None or source_img.size == 0:
        return None, None, None

    img_copy = source_img.copy()

    imgGrayscale, imgThresh = preprocess(img_copy)

    canny_image = cv2.Canny(imgThresh, 250, 255)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=2)

    linesP = Hough_transform(dilated_image, nol=8)

    if linesP is None:
        return 0, imgThresh, source_img

    angle = rotation_angle(linesP) / 2 * 1.5

    rotated_thresh = rotate_image(imgThresh, angle)
    rotated_img = rotate_image(source_img, angle)

    final_img = auto_crop_license_plate(rotated_img, rotated_thresh)

    return angle, rotated_thresh, final_img


if __name__ == "__main__":
    img = cv2.imread("dataset/cropped/greenpack_0013_0.jpg")

    if img is not None:
        angle, binary_img, rotated_lp = auto_rotate_and_crop_lp(img)

        if rotated_lp is not None:
            print(f"Rotation angle: {angle:.2f} degrees")
            cv2.imshow("Original", img)
            cv2.imshow("Rotated License Plate", rotated_lp)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Failed to load image")
