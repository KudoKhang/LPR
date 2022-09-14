from utilss import *
# Init models
model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/plate_yolo10k.pt', force_reload=True)
model_detect_character = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/character_yolo_087.pt')
model_recognize_character = CNN_Model(trainable=False).model
model_recognize_character.load_weights('src/weights/classify_character.h5')

def detect_plate(img):
    results = model_detect(img)
    t = results.pandas().xyxy[0]
    if len(t) > 0:
        # TODO: check area and confident
        bbox = np.int32(np.array(t)[:, :4][np.argmax(np.array(t)[:, 4])]) # Max confident
        # bbox = np.int32(np.array(t)[:, :4][np.argmax((np.array(t)[:, 2] - np.array(t)[:, 0]) * (np.array(t)[:, 3] - np.array(t)[:, 1]))]) # Max area
        plate = crop(img, bbox)
        return plate, bbox
    else:
        return None, None

def detect_char(plate, show_binary=False):
    condidates = []
    condidates_for_visualize = []
    results = model_detect_character(plate)
    t = results.pandas().xyxy[0]

    # Take the 8 characters with the highest confidence score
    bbox = np.int32(np.array(t)[:,:5][np.where(np.array(t)[:,4] > 0.5)]).tolist()
    bbox.sort(key=lambda x:x[4], reverse=True)
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

def recognize_char(candidates):
    characters = []
    coordinates = []

    for char, coordinate in candidates:
        characters.append(char)
        coordinates.append(coordinate)

    characters = np.array(characters)
    result = model_recognize_character.predict_on_batch(characters)
    result_idx = np.argmax(result, axis=1)

    candidates = []
    for i in range(len(result_idx)):
        if result_idx[i] == 31:  # if is background or noise, ignore it
            # If idx == 31, expand area bbox --> 110 % -- 78C00149(2).jpg
            char_err = characters[i]
            continue
        candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    return candidates

def E2E(image):
    plate, bbox = detect_plate(image)

    plate = automatic_brightness_and_contrast(plate)

    # plate = transform_plate(plate)

    if plate is None:
        return image, 'No License Plate Detected!'

    candidates_binary, h_avg = detect_char(plate, show_binary=True)
    candidates_predict = recognize_char(candidates_binary)
    license_plate = format(candidates_predict, h_avg)
    img = draw_labels_and_boxes(image, license_plate, bbox)
    return img, license_plate

def eval(root='../private_test/BAD/'):
    remove_space(root) # 76C12345 (2).jpg --> 76C12345.jpg
    path_image = [name for name in os.listdir(root) if name.endswith('jpg')]
    true, total_image= 0, len(path_image)

    os.makedirs('tests/', exist_ok=True)
    err_log = 'tests/error_val.txt'
    BoG = root.split('/')[-2]
    log = f'tests/log_{BoG}.txt'

    if os.path.exists(err_log):
        os.remove(err_log)
    if os.path.exists(log):
        os.remove(log)

    for path in path_image:
        try:
            label = path.split('(')[0] if '(' in path else path.split('.')[0] # Case 74C12355.jpg or 74C12355(1).jpg
            img = cv2.imread(os.path.join(root, path))
            img, license_plate = E2E(img)
            predict = ''.join(license_plate.split('-'))
            if  predict == label:
                true += 1
            else:
                print(f"Image: {path: <16} ---Failed--- Ground Trust: {label: <10} {'-': <3} Predict: {predict}")
                with open(log, 'a') as f:
                    f.write(f"Image: {path: <16} ---Failed--- Ground Trust: {label: <10} {'-': <3} Predict: {predict} \n")
                    f.close()
        except:
            print(f"Error: {path}")
            total_image -= 1
            with open(err_log, 'a') as f:
                f.write(f"{path} \n")
                f.close()

    content = f"{'-' * 90} \nAccuracy of private test --{root}-- | {true} / {total_image} = {round(true * 100 / total_image, 2)} %\nTotal error image is: {get_num_error(err_log)}. See logs here 👉 ---{err_log}---"
    print(content)
    with open(log, 'a') as f:
        f.write(content)
        f.close()

def process_folder(path_folder='data/pravite_test_500/', output_folder='output/results_clean/'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    image_path = [name for name in os.listdir(path_folder) if name.endswith(('jpg', 'png'))]
    err_log = 'tests/error.txt'
    if os.path.exists(err_log):
        os.remove(err_log)
    for path in tqdm(image_path):
        try:
            img = cv2.imread(os.path.join(path_folder, path))
            img, license_plate = E2E(img)
            cv2.imwrite(os.path.join(output_folder, path), img)
        except:
            with open(err_log, 'a') as f:
                f.write(f"{path} \n")
                f.close()
            print(f"Error: {path}")
    print(f"\nTotal error image is: {get_num_error(err_log)}. See logs here 👉 ---{err_log}---")

def process_image(image_path, show_image=False):
    img = cv2.imread(image_path)
    img, license_plate = E2E(img)
    print(license_plate)
    if show_image:
        cv2.imshow('Result LPR Predict', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    # process_folder('data/private_test/BAD/', 'output/private_test/BAD/')
    start = time.time()
    process_image('data/private_test/GOOD/92C14796.jpg')
    print(time.time() - start)
    # eval('./data/private_test/GOOD/')

"""
    transform: 77C11712.jpg
"""