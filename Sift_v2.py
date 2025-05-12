import numpy as np
import cv2
import math
from typing import List, Tuple, Optional
import time
import os

SIGMA = 1.6
ASSUMED_BLUR = 0.5
IMAGES = 5
SCALES = IMAGES - 3
BORDER = 1
CONTRAST_THRESHOLD = 0.04
EIGEN_VALUE_RATIO = 10
SCALE_FACTOR = 1.5
RADIUS_FACTOR = 3.0
BINS = 36
PEAK_RATIO = 0.8
SCALE_MULTIPLIER = 3.0
WINDOW_WIDTH = 4
DESCRIPTOR_MAX = 0.2
PI = math.pi

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

def rotate_image(image, angle=90):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))

    return rotated

class SiftHandler:

    def __init__(self, name: str, base_image):
        self.name = name
        self.base_image=base_image
        temp = cv2.cvtColor(self.base_image, cv2.COLOR_BGR2GRAY)
        self.temp = temp.astype(np.float64)
        self.onex = temp.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        temp = clahe.apply(temp)

        height, width = temp.shape
        self.octaves = int(round(math.log2(min(height, width))))-2

        print(f"Octave: {self.octaves}")
        print(f"Size: {width}x{height}")

        interpolated = cv2.resize(self.temp, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
        diff = max(math.sqrt(SIGMA ** 2 - 4 * ASSUMED_BLUR ** 2), 0.1)

        self.base = cv2.GaussianBlur(interpolated, (0, 0), diff)
        self.gauss_images = []
        self.images = []
        self.keypoints = []
        self.descriptors = []


    @timeit
    def gen_gaussian_images(self):

        k = 2 ** (1.0 / SCALES)
        kernel = [0] * IMAGES
        kernel[0] = SIGMA
        prev = SIGMA
        for i in range(1, IMAGES):
            now = prev * k
            kernel[i] = math.sqrt(now ** 2 - prev ** 2)
            prev = now
        temp = self.base.copy()
        self.gauss_images = []
        for i in range(self.octaves):
            octave_images = [None] * IMAGES
            octave_images[0] = temp
            for j in range(1, IMAGES):
                octave_images[j] = cv2.GaussianBlur(octave_images[j - 1], (0, 0), kernel[j])
            baseid = IMAGES - 3
            temp = cv2.resize(octave_images[baseid], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            self.gauss_images.append(octave_images)

    @timeit
    def gen_dog_images(self):
        self.images = []
        for octave in self.gauss_images:
            dog_images = [None] * (IMAGES - 1)
            for j in range(1, IMAGES):
                dog_images[j - 1] = cv2.subtract(octave[j], octave[j - 1])
            self.images.append(dog_images)

    def get_pixel_cube(self, oct: int, img: int, i: int, j: int) -> List[np.ndarray]:

        first_image = self.images[oct][img - 1]
        second_image = self.images[oct][img]
        third_image = self.images[oct][img + 1]
        pixel_cube = [
            first_image[i - 1:i + 2, j - 1:j + 2].copy(),
            second_image[i - 1:i + 2, j - 1:j + 2].copy(),
            third_image[i - 1:i + 2, j - 1:j + 2].copy()
        ]
        return pixel_cube

    def is_pixel_extremum(self, pixel_cube: List[np.ndarray]) -> bool:
        is_maximum = True
        is_minimum = True
        threshold = np.floor(0.5 * CONTRAST_THRESHOLD / SCALES * 255)
        center_value = pixel_cube[1][1, 1]

        if abs(center_value) <= threshold:
            return False

        for k in range(3):
            for i in range(3):
                for j in range(3):
                    if k == 1 and i == 1 and j == 1:
                        continue
                    is_maximum &= center_value >= pixel_cube[k][i, j]
                    is_minimum &= center_value <= pixel_cube[k][i, j]

        return ((center_value < 0) and is_minimum) or ((center_value > 0) and is_maximum)

    def get_gradient(self, pixel_cube: List[np.ndarray]) -> np.ndarray:

        grad = np.zeros((3, 1), dtype=np.float64)
        grad[0, 0] = 0.5 * (pixel_cube[1][1, 2] - pixel_cube[1][1, 0])
        grad[1, 0] = 0.5 * (pixel_cube[1][2, 1] - pixel_cube[1][0, 1])
        grad[2, 0] = 0.5 * (pixel_cube[2][1, 1] - pixel_cube[0][1, 1])

        return grad

    def get_hessian(self, pixel_cube: List[np.ndarray]) -> np.ndarray:

        hess = np.zeros((3, 3), dtype=np.float64)
        hess[0, 0] = pixel_cube[1][1, 2] - 2 * pixel_cube[1][1, 1] + pixel_cube[1][1, 0]
        hess[1, 1] = pixel_cube[1][2, 1] - 2 * pixel_cube[1][1, 1] + pixel_cube[1][0, 1]
        hess[2, 2] = pixel_cube[2][1, 1] - 2 * pixel_cube[1][1, 1] + pixel_cube[0][1, 1]

        hess[0, 1] = hess[1, 0] = 0.25 * (
                pixel_cube[1][2, 2] - pixel_cube[1][2, 0] - pixel_cube[1][0, 2] + pixel_cube[1][0, 0]
        )
        hess[0, 2] = hess[2, 0] = 0.25 * (
                pixel_cube[2][1, 2] - pixel_cube[2][1, 0] - pixel_cube[0][1, 2] + pixel_cube[0][1, 0]
        )
        hess[1, 2] = hess[2, 1] = 0.25 * (
                pixel_cube[2][2, 1] - pixel_cube[2][0, 1] - pixel_cube[0][2, 1] + pixel_cube[0][0, 1]
        )
        return hess

    def localize_extrema(self, oct: int, img: int, i: int, j: int) -> Tuple[int, Optional[dict]]:
        attempts = 5
        height, width = self.images[oct][0].shape

        for attempt in range(attempts):
            pixel_cube = self.get_pixel_cube(oct, img, i, j)

            for k in range(3):
                pixel_cube[k] = pixel_cube[k] / 255.0

            grad = self.get_gradient(pixel_cube)

            hess = self.get_hessian(pixel_cube)

            try:
                res = -np.linalg.solve(hess, grad) # H⋅x=grad => x
            except np.linalg.LinAlgError:
                return -1, None

            if abs(res[0, 0]) < 0.5 and abs(res[1, 0]) < 0.5 and abs(res[2, 0]) < 0.5:
                break

            j += int(round(res[0, 0]))
            i += int(round(res[1, 0]))
            img += int(round(res[2, 0]))

            if (i < BORDER or i >= height - BORDER or
                    j < BORDER or j >= width - BORDER or
                    img < 1 or img > SCALES):
                return -1, None

            if attempt == attempts - 1 and (abs(res[0, 0]) >= 0.5 or abs(res[1, 0]) >= 0.5 or abs(res[2, 0]) >= 0.5):
                return -1, None

        value = pixel_cube[1][1, 1] + 0.5 * np.dot(grad.transpose(), res)[0, 0]

        if abs(value) * SCALES >= CONTRAST_THRESHOLD:
            hess2 = hess[:2, :2]
            hess_trace = np.trace(hess2)
            hess_det = np.linalg.det(hess2)

            if hess_det <= 0:
                return -1, None

            if (hess_trace ** 2) * EIGEN_VALUE_RATIO < (EIGEN_VALUE_RATIO + 1) ** 2 * hess_det:
                keypt_octave = oct + (1 << 8) * img + (1 << 16) * int(round((res[2, 0] + 0.5) * 255))
                keypt_pt_x = (j + res[0, 0]) * (1 << oct)
                keypt_pt_y = (i + res[1, 0]) * (1 << oct)
                keypt_size = SIGMA * (2 ** (img + res[2, 0] / SCALES)) * (1 << (oct + 1))
                keypt_response = abs(value)

                keypoint = {
                    'pt': (keypt_pt_x, keypt_pt_y),
                    'size': keypt_size,
                    'angle': -1.0,
                    'response': keypt_response,
                    'octave': keypt_octave
                }
                return img, keypoint

        return -1, None

    def get_keypoint_orientations(self, oct: int, img: int, keypoint: dict) -> List[dict]:
        height, width = self.gauss_images[oct][img].shape

        hist = [0.0] * BINS
        smooth = [0.0] * BINS

        base_x = int(round(keypoint['pt'][0] / (1 << oct)))
        base_y = int(round(keypoint['pt'][1] / (1 << oct)))

        scale = SCALE_FACTOR * keypoint['size'] / (1 << (oct + 1))
        radius = int(round(scale * RADIUS_FACTOR))
        weight_factor = -0.5 / (scale ** 2)

        for i in range(-radius, radius + 1):
            if 0 < base_y + i < height - 1:
                for j in range(-radius, radius + 1):
                    if 0 < base_x + j < width - 1:
                        dx = self.gauss_images[oct][img][base_y + i, base_x + j + 1] - \
                             self.gauss_images[oct][img][base_y + i, base_x + j - 1]
                        dy = self.gauss_images[oct][img][base_y + i - 1, base_x + j] - \
                             self.gauss_images[oct][img][base_y + i + 1, base_x + j]
                        mag = math.sqrt(dx ** 2 + dy ** 2)
                        orientation = math.degrees(math.atan2(dy, dx))
                        index = (int(round((orientation * BINS) / 360)) % BINS + BINS) % BINS
                        hist[index] += math.exp(weight_factor * (i ** 2 + j ** 2)) * mag

        for i in range(BINS):
            smooth[i] = (6 * hist[i] +
                         4 * (hist[(i - 1) % BINS] +  4 * hist[(i + 1) % BINS]) +
                         hist[(i - 2) % BINS] + hist[(i + 2) % BINS]) / 16.0

        max_orientation = max(smooth)
        new_keypoints = []

        for i in range(BINS):
            l = smooth[(i - 1) % BINS]
            r = smooth[(i + 1) % BINS]
            if smooth[i] > l and smooth[i] > r:
                peak = smooth[i]
                if peak >= PEAK_RATIO * max_orientation:
                    interpolated_index = (i + 0.5 * (l - r) / (l + r - 2 * peak)) % BINS
                    orientation = 360 - interpolated_index * 360 / BINS
                    if abs(360 - orientation) < 1e-7:
                        orientation = 0

                    new_keypoint = keypoint.copy()
                    new_keypoint['angle'] = orientation
                    new_keypoints.append(new_keypoint)

        return new_keypoints

    @timeit
    def gen_scale_space_extrema(self):
        self.keypoints = []

        for oct in range(self.octaves):
            for img in range(1, len(self.images[oct]) - 1):
                height, width = self.images[oct][img].shape

                for i in range(BORDER, height - BORDER):
                    for j in range(BORDER, width - BORDER):
                        pixel_cube = self.get_pixel_cube(oct, img, i, j)
                        if self.is_pixel_extremum(pixel_cube):
                            img_index, keypoint = self.localize_extrema(oct, img, i, j)
                            if img_index >= 0:
                                new_keypoints = self.get_keypoint_orientations(oct, img_index, keypoint)
                                self.keypoints.extend(new_keypoints)

    @timeit
    def clean_keypoints(self):
        if not self.keypoints:
            return

        self.keypoints.sort(key=lambda kp: (
            kp['pt'][0], kp['pt'][1], -kp['size'], kp['angle'],
            -kp['response'], -kp['octave'] & 255, -(kp['octave'] >> 8) & 255
        ))

        EPS2 = 1e-4
        unique_kpts = [self.keypoints[0]]

        for i in range(1, len(self.keypoints)):
            kp = self.keypoints[i]
            prev_kp = unique_kpts[-1]

            if (abs(kp['pt'][0] - prev_kp['pt'][0]) > EPS2 or
                    abs(kp['pt'][1] - prev_kp['pt'][1]) > EPS2 or
                    abs(kp['size'] - prev_kp['size']) > EPS2 or
                    abs(kp['angle'] - prev_kp['angle']) > EPS2):
                unique_kpts.append(kp)

        for kp in unique_kpts:
            kp['pt'] = (kp['pt'][0] * 0.5, kp['pt'][1] * 0.5)
            kp['size'] *= 0.5
            octave = kp['octave'] & 255
            new_octave = ((kp['octave'] & ~255) | ((octave - 1) & 255))
            kp['octave'] = new_octave

        self.keypoints = unique_kpts

    def get_descriptor(self, keypoint: dict) -> List[float]:

        octave = keypoint['octave'] & 255
        layer = (keypoint['octave'] >> 8) & 255

        if octave >= 128:
            octave |= -128

        scale = 1.0 / (1 << octave) if octave >= 0 else (1 << -octave)

        image = self.gauss_images[octave + 1][layer]
        height, width = image.shape

        pt_x = int(round(scale * keypoint['pt'][0]))
        pt_y = int(round(scale * keypoint['pt'][1]))
        angle = 360 - keypoint['angle']

        cos_angle = math.cos(math.radians(angle))
        sin_angle = math.sin(math.radians(angle))
        weight_multiplier = -0.5 / ((0.5 * WINDOW_WIDTH) ** 2)

        bins = 8
        bins_per_degree = bins / 360.0

        hist_width = SCALE_MULTIPLIER * 0.5 * scale * keypoint['size']

        half_width = min(
            int(round(hist_width * (WINDOW_WIDTH + 1) / math.sqrt(2.0))),
            int(math.sqrt(height ** 2 + width ** 2))
        )

        rows = []
        cols = []
        magnitudes = []
        orientations = []

        for i in range(-half_width, half_width + 1):
            for j in range(-half_width, half_width + 1):
                row_rotation = sin_angle * j + cos_angle * i
                col_rotation = sin_angle * i - cos_angle * j

                bin_row = (row_rotation / hist_width) + 0.5 * (WINDOW_WIDTH - 1)
                bin_col = (col_rotation / hist_width) + 0.5 * (WINDOW_WIDTH - 1)

                if -1 < bin_row < WINDOW_WIDTH and -1 < bin_col < WINDOW_WIDTH:
                    win_row = pt_y + i
                    win_col = pt_x + j

                    if 0 < win_row < height - 1 and 0 < win_col < width - 1:
                        dx = image[win_row, win_col + 1] - image[win_row, win_col - 1]
                        dy = image[win_row - 1, win_col] - image[win_row + 1, win_col]

                        mag = math.sqrt(dx ** 2 + dy ** 2)
                        orient = math.fmod(math.degrees(math.atan2(dy, dx)), 360.0)

                        exponent = (row_rotation / hist_width) ** 2 + (col_rotation / hist_width) ** 2
                        weight = math.exp(weight_multiplier * exponent)

                        rows.append(bin_row)
                        cols.append(bin_col)
                        magnitudes.append(mag * weight)
                        orientations.append((orient - angle) * bins_per_degree)

        tensor = np.zeros((WINDOW_WIDTH + 2, WINDOW_WIDTH + 2, bins), dtype=np.float64)

        for l in range(len(rows)):
            row_bin = int(math.floor(rows[l]))
            col_bin = int(math.floor(cols[l]))
            orient_bin = int(math.floor(orientations[l]))

            row_bin_pr = rows[l] - row_bin
            col_bin_pr = cols[l] - col_bin
            orient_bin_pr = orientations[l] - orient_bin

            if orient_bin < 0:
                orient_bin += bins
            if orient_bin >= bins:
                orient_bin -= bins

            for i in range(2):
                row_wt = (1 - row_bin_pr) if i == 0 else row_bin_pr
                for j in range(2):
                    col_wt = (1 - col_bin_pr) if j == 0 else col_bin_pr
                    for k in range(2):
                        orient_wt = (1 - orient_bin_pr) if k == 0 else orient_bin_pr
                        c = magnitudes[l] * row_wt * col_wt * orient_wt
                        tensor[row_bin + 1 + i, col_bin + 1 + j, (orient_bin + k) % bins] += c

        descriptor_vector = []

        for i in range(1, WINDOW_WIDTH + 1):
            for j in range(1, WINDOW_WIDTH + 1):
                for k in range(bins):
                    descriptor_vector.append(tensor[i, j, k])

        norm = math.sqrt(sum(d ** 2 for d in descriptor_vector))
        thresh = norm * DESCRIPTOR_MAX

        for i in range(len(descriptor_vector)):
            descriptor_vector[i] = min(descriptor_vector[i], thresh)

        norm = math.sqrt(sum(d ** 2 for d in descriptor_vector))
        norm = max(norm, 1e-7)

        for i in range(len(descriptor_vector)):
            descriptor_vector[i] = min(max(round(descriptor_vector[i] / norm * 512), 0), 255)

        return descriptor_vector

    @timeit
    def get_descriptors(self):
        self.descriptors = []
        for kp in self.keypoints:
            self.descriptors.append(self.get_descriptor(kp))

    def dump_keypoints(self):
        with open(f"{self.name}.csv", "w") as fout:
            for kp in self.keypoints:
                fout.write(f"{kp['pt'][0]},{kp['pt'][1]},{kp['size']},{kp['angle']}\n")

    def get_cv2_keypoints(self):
        cv_keypoints = []
        for kp in self.keypoints:
            cv_kp = cv2.KeyPoint(
                x=float(kp['pt'][0]),
                y=float(kp['pt'][1]),
                size=float(kp['size']),
                angle=float(kp['angle']),
                response=float(kp['response']),
                octave=int(kp['octave'])
            )
            cv_keypoints.append(cv_kp)
        return cv_keypoints

    def get(self):
        if not self.descriptors:
            return None

        desc = np.zeros((len(self.descriptors), len(self.descriptors[0])), dtype=np.float32)
        for i, descriptor in enumerate(self.descriptors):
            for j, value in enumerate(descriptor):
                desc[i, j] = value
        return desc

    def exec(self):
        print(f"Running SIFT for {self.name}")

        self.gen_gaussian_images()
        self.gen_dog_images()

        temp = self.onex.astype(np.uint8)
        temp2 = temp.copy()

        self.gen_scale_space_extrema()
        self.clean_keypoints()
        cv_keypoints = self.get_cv2_keypoints()
        #
        # out = cv2.drawKeypoints(temp2, cv_keypoints, None,
        #                         color=(0, 0, 255),
        #                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imwrite(f"{self.name}_keypoints.png", out)


        self.get_descriptors()
        print(f"Keypoints Calculated: {len(self.keypoints)}")
        print(f"Completed SIFT for {self.name}\n")

def match_keypoints_bf(desc1, desc2, threshold=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    desc1 = np.array(desc1, dtype=np.float32)
    desc2 = np.array(desc2, dtype=np.float32)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    matches = []
    best_matches = {}

    for m in raw_matches:
        if len(m) < 2:
            continue
        d0, d1 = m[0], m[1]
        if d0.distance < threshold * d1.distance:
            queryIdx = d0.queryIdx
            trainIdx = d0.trainIdx
            distance = d0.distance
            if trainIdx not in best_matches or distance < best_matches[trainIdx][0]:
                best_matches[trainIdx] = (distance, queryIdx)

    for trainIdx, (distance, queryIdx) in best_matches.items():
        matches.append(( queryIdx,trainIdx))

    return matches


def match_keypoints_flann(desc1, desc2, threshold=0.75):
    desc1 = np.array(desc1, dtype=np.float32)
    desc2 = np.array(desc2, dtype=np.float32)

    # FLANN parameters cho SIFT/ORB (SIFT dùng L2)
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(desc1, desc2, k=2)

    matches = []
    best_matches = {}

    for m in raw_matches:
        if len(m) < 2:
            continue
        d0, d1 = m[0], m[1]
        if d0.distance < threshold * d1.distance:
            queryIdx = d0.queryIdx
            trainIdx = d0.trainIdx
            distance = d0.distance
            if trainIdx not in best_matches or distance < best_matches[trainIdx][0]:
                best_matches[trainIdx] = (distance, queryIdx)

    for trainIdx, (distance, queryIdx) in best_matches.items():
        matches.append((queryIdx, trainIdx))

    return matches
def draw_matches(img1, keypoints1, img2, keypoints2, matches):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    matched_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    matched_img[:h1, :w1] = img1
    matched_img[:h2, w1:] = img2

    for (i1, i2) in matches:
        x1, y1 = keypoints1[i1]['pt']
        x2, y2 = keypoints2[i2]['pt']

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + w1, int(y2))

        cv2.circle(matched_img, pt1, 1, (0, 255, 0), -1)
        cv2.circle(matched_img, pt2, 1, (0, 255, 0), -1)

        cv2.line(matched_img, pt1, pt2, (255, 0, 0), 1)

    return matched_img
def draw_bounding_box_on_img2( img2, keypoints2, matches):
    if(len(matches)==0): return img2,None
    pts2 = np.float32([ (kp['pt'][0],kp['pt'][1]) for _, j in matches for kp in [keypoints2[j]]])  # (x, y)

    expand = 6
    min_x = max(int(np.floor(np.min(pts2[:, 0])) - expand), 0)
    max_x = min(int(np.ceil(np.max(pts2[:, 0])) + expand), img2.shape[1] - 1)
    min_y = max(int(np.floor(np.min(pts2[:, 1])) - expand), 0)
    max_y = min(int(np.ceil(np.max(pts2[:, 1])) + expand), img2.shape[0] - 1)

    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()

    cv2.rectangle(img2_rgb, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    bbox = (min_x, min_y, max_x, max_y)
    cropped_region = img2[min_y:max_y+1, min_x:max_x+1]
    return img2_rgb, cropped_region,bbox
def crop_bounding_box_from_img2(img2, bbox):
    if bbox is None:
        return None

    min_x, min_y, max_x, max_y = bbox
    cropped = img2[min_y:max_y+1, min_x:max_x+1]
    return cropped
def draw_keypoints(image, keypoints):
    img_with_keypoints = image.copy()

    for keypoint in keypoints:
        x,y=keypoint['pt']
        cv2.circle(img_with_keypoints, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=1)
    return img_with_keypoints
def main():

    image_path = "dataset/images/greenpack_0005.png"
    if os.path.exists(image_path):
        img1 = cv2.imread(image_path)
        if img1 is not None:
            sift1 = SiftHandler("example1", img1)
            sift1.exec()

    image_path = "dataset/images/greenpack_0005.png"

    if os.path.exists(image_path):
        img2 = cv2.imread(image_path)

        img2=cv2.GaussianBlur(img2,(0,0),5)
        if img2 is not None:
            sift2 = SiftHandler("example2", img2)
            sift2.exec()
    cv2.imshow("1", draw_keypoints(sift1.base_image, sift1.keypoints))
    cv2.imshow("2", draw_keypoints(sift2.base_image, sift2.keypoints))

    matches=match_keypoints_bf(sift1.descriptors,sift2.descriptors)
    matched_img = draw_matches(sift1.base_image, sift1.keypoints, sift2.base_image, sift2.keypoints, matches)
    cv2.imshow("Keypoints Matching", matched_img)
    result,bbox = draw_bounding_box_on_img2(sift2.base_image, sift2.keypoints, matches)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()