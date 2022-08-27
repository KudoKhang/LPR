import os.path

from .libs import *

def get_max_area(bbox):
    if len(bbox) == 1:
        return bbox[0]
    if len(bbox) == 0:
        return "No plate detection!"
    else:
        area = [(bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])]
        index_max = int(np.where(max(area))[0])
        return bbox[index_max]

def segmentation(LpRegion):
    candidates = []
    # apply thresh to extracted licences plate
    V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]

    # adaptive threshold
    T = threshold_local(V, 15, offset=10, method="gaussian")
    thresh = (V > T).astype("uint8") * 255

    # convert black pixel of digits to white pixel
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, width=400)
    thresh = cv2.medianBlur(thresh, 5)

    # connected components analysis
    labels = measure.label(thresh, connectivity=2, background=0)

    # loop over the unique components
    for label in np.unique(labels):
        # if this is background label, ignore it
        if label == 0:
            continue

        # init mask to store the location of the character candidates
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

        # find contours from mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(contour)

            # rule to determine characters
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            heightRatio = h / float(LpRegion.shape[0])

            if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                # extract characters
                candidate = np.array(mask[y:y + h, x:x + w])
                square_candidate = convert2Square(candidate)
                square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                square_candidate = square_candidate.reshape((28, 28, 1))
                candidates.append((square_candidate, (y, x)))
    return candidates

def detect_char_rgb(lpRegion):
    condidates = []
    results = model_detect_character(lpRegion)
    t = results.pandas().xyxy[0]
    bbox = list(np.int32(np.array(t)[:, :4]))
    bbox.sort(key=lambda x: x[0])

    if len(bbox) > 0:
        for bb in bbox:
            x1, y1, x2, y2 = bb
            cv2.rectangle(lpRegion, (x1,y1), (x2,y2), (0,255,0), 1)
            char = lpRegion.copy()[y1:y2, x1:x2]

            bg = np.zeros((char.shape[0], char.shape[0]))
            x = int((char.shape[0] - char.shape[1]) / 2)

            char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
            bg[0:char.shape[0], x: x + char.shape[1]] = char
            bg = cv2.resize(bg, (28, 28))

            condidates.append((bg, (y1, x1)))
    return condidates


def convert2Square(image):
    img_h = image.shape[0]
    img_w = image.shape[1]

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff // 2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff // 2))
            x2 = np.zeros(shape=(img_h, (diff // 2) + 1))
        squared_image = np.concatenate((x1, image, x2), axis=1)

    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff // 2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff // 2, img_w))
            x2 = x1
        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image
