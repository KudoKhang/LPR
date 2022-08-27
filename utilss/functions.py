import os.path

from .libs import *


ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

def crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def get_num_error(path_err):
    # Return num of error images in logs of eval function
    if not os.path.exists(path_err):
        return 0
    with open(path_err, 'r') as f:
        num_lines = sum(1 for line in f)
        return num_lines

def draw_labels_and_boxes(image, labels, boxes):
    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[2])
    y_max = round(boxes[3])

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    image = cv2.putText(image, labels, (x_min - 40, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

    return image

def remove_space(root):
    # 76C12345 (2).jpg --> 76C12345(2).jpg
    path_image = [name for name in os.listdir(root) if name.endswith('jpg')]
    for path in tqdm(path_image):
        new_name = ''.join(path.split())
        os.rename(os.path.join(root + path), os.path.join(root + new_name))

#------------------------------------------------------------------------------------------------------------

def try_catch(line1, line2):
    index = len(line1)
    line = line1 + line2

    dict_try_catch_line1 = {'0': 'C'}
    dict_try_catch_line2 = {'D': '0',
                            'C': '0',
                            'B': '8'}

    for i in range(len(line)):
        if i == 2:
            # Some case classify character to noise --> id == 2 not right
            continue

        for key in dict_try_catch_line2.keys():
            if line[i][0] == key:
                temp = list(line[i])
                temp[0] = dict_try_catch_line2[key]
                line[i] = tuple(temp)

    return line[:index], line[index:]

def padding(thresh, h=400):
    # Denoise -> padding to sqare -> resize to (28x28) -> forward to model classify
    char_origin = denoise(thresh)
    thresh = imutils.resize(thresh, height=h)
    thresh = cv2.medianBlur(thresh, 5)

    char_bg = np.zeros((h, h))
    x = int((h - thresh.shape[1]) / 2)
    char_bg[0:h, x: x + thresh.shape[1]] = thresh
    char_bg = denoise(char_bg)
    char_bg = cv2.resize(char_bg, (28, 28))
    return char_bg, char_origin

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
            first_line.append((candidate, coordinate[1]))
        else:
            second_line.append((candidate, coordinate[1]))

    def take_second(s):
        return s[1]

    first_line = sorted(first_line, key=take_second)
    second_line = sorted(second_line, key=take_second)

    # Catch some case confuse
    first_line, second_line = try_catch(first_line, second_line)

    if len(second_line) == 0:
        license_plate = "".join([str(ele[0]) for ele in first_line])
        license_plate = license_plate[:3] + '-' + license_plate[3:]
    else:
        license_plate = "".join([str(ele[0]) for ele in second_line]) + "-" +  "".join([str(ele[0]) for ele in first_line])

    return license_plate

def denoise(char):
    # Remove noise in character binary via findContour --> calculate Area --> Compare with threshold area
    char = np.uint8(char)
    _, mask = cv2.threshold(char, 15, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lst_coord_and_area = []
    for c in contours:
        area =cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        lst_coord_and_area.append((x, y , w, h, area))

    if len(lst_coord_and_area) > 0:
        lst_coord_and_area.pop(np.argmax(np.array(lst_coord_and_area)[:, -1]))

    for bb in lst_coord_and_area:
        x, y, w, h = bb[:4]
        points = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        cv2.fillPoly(char, pts=[points], color=(0, 0, 0))

    return char

def transform_plate(img):
    w, h = img.shape[:2]

    pt_A = [6, 2]
    pt_B = [3, 47]
    pt_C = [56, 56]
    pt_D = [62, 12]

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])

    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    img = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return img

# ------------------------------------------------------------------------------------------------------------
