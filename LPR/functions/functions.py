import math
import os
import os.path
import warnings

import cv2
import imutils
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

ALPHA_DICT = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "R",
    14: "S",
    15: "T",
    16: "U",
    17: "V",
    18: "X",
    19: "Y",
    20: "Z",
    21: "0",
    22: "1",
    23: "2",
    24: "3",
    25: "4",
    26: "5",
    27: "6",
    28: "7",
    29: "8",
    30: "9",
    31: "Background",
}


def draw(img, bbox):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)


def get_center(bbox):
    cx, cy = bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2
    return [cx, cy]


def crop(image, bbox):
    return image[bbox[1] : bbox[3], bbox[0] : bbox[2]]


def get_num_error(path_err):
    # Return num of error images in logs of eval function
    if not os.path.exists(path_err):
        return 0
    with open(path_err, "r") as f:
        num_lines = sum(1 for line in f)
        return num_lines


def draw_corner(img, t):
    # bbox = np.int32(np.array(t)[:, :4][np.where(np.array(t)[:, 4] > 0.6)]
    bbox = np.int32(np.array(t)[:, :4])
    for bb in bbox:
        draw(img, bb)


def draw_labels_and_boxes(image, labels, boxes):
    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[2])
    y_max = round(boxes[3])

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    image = cv2.putText(
        image,
        labels,
        (x_min - 40, y_min),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
    )

    return image


def remove_space(root):
    # 76C12345 (2).jpg --> 76C12345(2).jpg
    path_image = [name for name in os.listdir(root) if name.endswith("jpg")]
    for path in tqdm(path_image):
        new_name = "".join(path.split())
        os.rename(os.path.join(root + path), os.path.join(root + new_name))


# ------------------------------------------------------------------------------------------------------------


def try_catch(line1, line2):
    index = len(line1)
    dict_try_catch_line1 = {
        "0": "C",
        "6": "C",
        "L": "C",
        "2": "C",
        "Z": "C",
        "F": "C",
        "8": "C",
    }

    dict_try_catch_line2 = {
        "D": "0",
        "C": "0",
        "B": "8",
        "E": "8",
        "A": "8",
        "H": "8",
        "K": "4",
    }
    line = line1 + line2

    for i in range(len(line)):
        if i == 2 and len(line) > 6:
            for key in dict_try_catch_line1.keys():
                if line[i][0] == key:
                    temp = list(line[i])
                    temp[0] = dict_try_catch_line1[key]
                    line[i] = tuple(temp)
        else:
            for key in dict_try_catch_line2.keys():
                if line[i][0] == key:
                    temp = list(line[i])
                    temp[0] = dict_try_catch_line2[key]
                    line[i] = tuple(temp)

    return line[:index], line[index:]


def draw_bbox_character(plate, bbox):
    for bb in bbox:
        x1, y1, x2, y2 = bb
        cv2.rectangle(plate, (x1, y1), (x2, y2), (0, 255, 0), 1)


def padding(thresh, h=28):
    char_bg = np.zeros((h, h))
    x = int((h - thresh.shape[1]) / 2)
    char_bg[0:h, x : x + thresh.shape[1]] = thresh
    return char_bg


def binary_image(char):
    h, w = char.shape[:2]
    char = imutils.resize(char, height=28)
    char_gray = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
    char_blur = cv2.GaussianBlur(char_gray, (1, 1), 1)
    _, thresh = cv2.threshold(char_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)
    thresh_inv = denoise(thresh_inv)
    thresh_ori = cv2.resize(thresh_inv, (w, h))
    return thresh_inv, thresh_ori


def format(candidates, h_avg):
    first_line = []
    second_line = []

    # Get y_max in all characters
    lst_temp = []
    for l_sub in np.array(candidates, dtype=object)[:, 1]:
        lst_temp.append(list(l_sub))
    y_max = max(np.array(lst_temp)[:, 0])

    # Determined character to line1 or line2 by compare y_max with h_avg (height average character)
    for candidate, coordinate in candidates:
        if coordinate[0] + 0.75 * h_avg > y_max:
            second_line.append((candidate, coordinate[1]))
        else:
            first_line.append((candidate, coordinate[1]))

    def take_second(s):
        return s[1]

    first_line = sorted(first_line, key=take_second)
    second_line = sorted(second_line, key=take_second)

    # Catch some case confuse
    first_line, second_line = try_catch(first_line, second_line)

    if len(second_line) == 0:
        license_plate = "".join([str(ele[0]) for ele in first_line])
    else:
        license_plate = "".join([str(ele[0]) for ele in first_line]) + "".join([str(ele[0]) for ele in second_line])

    return license_plate


def denoise(char):
    # Remove noise in character binary via findContour --> calculate Area --> Compare with threshold area
    char = np.uint8(char)
    _, mask = cv2.threshold(char, 15, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lst_coord_and_area = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        lst_coord_and_area.append((x, y, w, h, area))

    if len(lst_coord_and_area) > 0:
        lst_coord_and_area.pop(np.argmax(np.array(lst_coord_and_area)[:, -1]))

    for bb in lst_coord_and_area:
        x, y, w, h = bb[:4]
        points = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        cv2.fillPoly(char, pts=[points], color=(0, 0, 0))

    return char


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= maximum / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


def transform_plate(img, pt_A, pt_B, pt_C, pt_D):
    # A-B-C-D : counter-clockwise
    w, h = img.shape[:2]

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]])

    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    img = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return img


def interpolate_end_point(A, B, C, D, coefficient_expand=1):
    # Tính chất hình bình hành: vector cặp cạnh đối diện luôn bằng nhau
    if len(A) == 0:
        x = B[0][0] + D[0][0] - C[0][0]
        y = B[0][1] + D[0][1] - C[0][1]
        A = [[x, y]]

    if len(B) == 0:
        x = A[0][0] + C[0][0] - D[0][0]
        y = A[0][1] + C[0][1] - D[0][1]
        B = [[x, y]]

    if len(C) == 0:
        x = B[0][0] + D[0][0] - A[0][0]
        y = B[0][1] + D[0][1] - A[0][1]
        C = [[x, y]]

    if len(D) == 0:
        x = A[0][0] + C[0][0] - B[0][0]
        y = A[0][1] + C[0][1] - B[0][1]
        D = [[x, y]]

    return A[0], B[0], C[0], D[0]


def get_true_coord(A, B, C, D, bbox_plate):
    w, h = bbox_plate[2] - bbox_plate[0], bbox_plate[3] - bbox_plate[1]

    _dict = {"A": (A, (0, 0)), "B": (B, (0, h)), "C": (C, (w, h)), "D": (D, (w, 0))}

    for key in _dict.keys():
        if len(_dict[key][0]) > 1:
            temp = []
            for center in _dict[key][0]:
                dist = math.dist(center, _dict[key][1])
                temp.append((dist, center))
            temp = np.array(temp)
            _dict[key] = (temp[:, 1][np.where(np.argmax(temp[:, 0]))], "000")

    return interpolate_end_point(A, B, C, D)


def ABCD(bbox, img, bbox_plate):
    h, w = img.shape[:2]
    A = []
    B = []
    C = []
    D = []
    for bb in bbox:
        center = get_center(bb)
        if center[0] < w / 2:
            A.append(center) if center[1] < h / 2 else B.append(center)
        else:
            D.append(center) if center[1] < h / 2 else C.append(center)

    return get_true_coord(A, B, C, D, bbox_plate)


def expand_bbox(bbox, img, scale=0.1):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    h_expand = int((y2 - y1) * scale)

    x1_exp = 0 if (x1 - h_expand) < 0 else (x1 - h_expand)
    y1_exp = 0 if (y1 - h_expand) < 0 else (y1 - h_expand)
    x2_exp = W if (x2 + h_expand) > W else (x2 + h_expand)
    y2_exp = H if (y2 - h_expand) > H else (y2 + h_expand)

    return (x1_exp, y1_exp, x2_exp, y2_exp)


# ------------------------------------------------------------------------------------------------------------
