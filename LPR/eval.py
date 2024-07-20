import os

import cv2
from tqdm import tqdm

from lpr.functions import core
from lpr.main import LicensePlateRecognition
from lpr.utils.logger import logger

model = LicensePlateRecognition()


def eval(root="data/private_test/BAD/"):
    core.remove_space(root)  # 76C12345 (2).jpg --> 76C12345.jpg
    path_image = [name for name in os.listdir(root) if name.endswith("jpg")]
    true, total_image = 0, len(path_image)

    os.makedirs("logs/", exist_ok=True)
    err_log = "logs/error_val.txt"
    BoG = root.split("/")[-2]
    log = f"logs/log_{BoG}.txt"

    if os.path.exists(err_log):
        os.remove(err_log)
    if os.path.exists(log):
        os.remove(log)

    for path in path_image:
        try:
            label = path.split("(")[0] if "(" in path else path.split(".")[0]  # Case 74C12355.jpg or 74C12355(1).jpg
            img = cv2.imread(os.path.join(root, path))
            license_plate, _ = model.E2E(img)
            predict = "".join(license_plate.split("-"))
            if predict == label:
                true += 1
            else:
                print(f"Image: {path: <16} ---Failed--- Ground Trust: {label: <10} {'-': <3} Predict: {predict}")
                with open(log, "a") as f:
                    f.write(f"Image: {path: <16} ---Failed--- Ground Trust: {label: <10} {'-': <3} Predict: {predict} \n")
                    f.close()
        except Exception as e:
            logger.warning(f"Error: {path}, error: {e}")
            total_image -= 1
            with open(err_log, "a") as f:
                f.write(f"{path} \n")
                f.close()

    content = f"{'-' * 90} \nAccuracy of private test --{root}-- | {true} / {total_image} = {round(true * 100 / total_image, 2)} %\nTotal error image is: {core.get_num_error(err_log)}. See logs here 👉 ---{err_log}---"
    print(content)
    with open(log, "a") as f:
        f.write(content)
        f.close()


def process_folder(path_folder="data/pravite_test/GOOD/", output_folder="output/results_clean/"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    image_path = [name for name in os.listdir(path_folder) if name.endswith(("jpg", "png"))]
    err_log = "tests/error.txt"
    if os.path.exists(err_log):
        os.remove(err_log)
    for path in tqdm(image_path):
        try:
            img = cv2.imread(os.path.join(path_folder, path))
            license_plate, _ = model.E2E(img)
            cv2.imwrite(os.path.join(output_folder, path), img)
        except Exception as e:
            with open(err_log, "a") as f:
                f.write(f"{path} \n")
                f.close()
            logger.error(f"Error: {path}, error: {e}")
    print(f"\nTotal error image is: {core.get_num_error(err_log)}. See logs here 👉 ---{err_log}---")


def process_image(image_path):
    img = cv2.imread(image_path)
    img, license_plate = model.E2E(img)
    print(license_plate)
    cv2.imshow("Result LPR Predict", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    eval()
