from utilss import *

class LicensePlateRecognition:
    def __init__(self, weight_plate='src/weights/plate_yolo10k.pt', weight_character='src/weights/character_yolo_087.pt', weight_classify='src/weights/classify_character.h5'):
        self.model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_plate,
                                      verbose=False)
        self.model_detect_character = torch.hub.load('ultralytics/yolov5', 'custom',
                                                path=weight_character, verbose=False)
        self.model_recognize_character = CNN_Model(trainable=False).model
        self.model_recognize_character.load_weights(weight_classify)

    def detect_plate(self, img):
        results = self.model_detect(img)
        t = results.pandas().xyxy[0]
        if len(t) > 0:
            # TODO: check area and confident
            bbox = np.int32(np.array(t)[:, :4][np.argmax(np.array(t)[:, 4])])  # Max confident
            # bbox = np.int32(np.array(t)[:, :4][np.argmax((np.array(t)[:, 2] - np.array(t)[:, 0]) * (np.array(t)[:, 3] - np.array(t)[:, 1]))]) # Max area
            plate = crop(img, bbox)
            return plate, bbox
        else:
            return None, None

    def detect_char(self, plate, show_binary=False):
        condidates = []
        condidates_for_visualize = []
        results = self.model_detect_character(plate)
        t = results.pandas().xyxy[0]

        # Take the 8 characters with the highest confidence score
        bbox = np.int32(np.array(t)[:, :5][np.where(np.array(t)[:, 4] > 0.5)]).tolist()
        bbox.sort(key=lambda x: x[4], reverse=True)
        bbox = np.array(bbox[:8])[:, :4].tolist()

        height_char = []

        if len(bbox) > 0:
            for bb in bbox:
                x1, y1, x2, y2 = bb
                height_char.append(y2 - y1)
                char = plate.copy()[y1:y2, x1:x2]

                thresh, thresh_ori = binary_image(char)

                character = padding(thresh)
                condidates.append((character, (y1, x1)))
                condidates_for_visualize.append((thresh_ori, (y1, x1)))

            # draw_bbox_character(plate, bbox)

            if show_binary:
                bg = np.zeros(plate.shape[:2])
                for c_b in condidates_for_visualize:
                    y, x = c_b[1]
                    binary = c_b[0]
                    h, w = binary.shape[:2]
                    bg[y:y + h, x:x + w] = binary
                cv2.imshow('Binary', cv2.resize(bg, tuple([a * 3 for a in bg.shape[::-1]])))

            return condidates, sum(height_char) / len(height_char)

        # TODO: Case BBox empty
        return 0, 0

    def recognize_char(self, candidates):
        characters = []
        coordinates = []

        for char, coordinate in candidates:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)
        result = self.model_recognize_character.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)

        candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31:  # if is background or noise, ignore it
                # If idx == 31, expand area bbox --> 110 % -- 78C00149(2).jpg
                char_err = characters[i]
                continue
            candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

        return candidates

    def E2E(self, image):
        plate, bbox = self.detect_plate(image)

        plate = automatic_brightness_and_contrast(plate)

        # plate = transform_plate(plate)

        if plate is None:
            return image, 'No License Plate Detected!'

        candidates_binary, h_avg = self.detect_char(plate, show_binary=False)
        candidates_predict = self.recognize_char(candidates_binary)
        license_plate = format(candidates_predict, h_avg)
        # img = draw_labels_and_boxes(image, license_plate, bbox)
        return image, license_plate, bbox

    def predict(self, image_path):
        # https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi
        img, license_plate, bbox = self.E2E(image_path)
        return license_plate, str(bbox)
