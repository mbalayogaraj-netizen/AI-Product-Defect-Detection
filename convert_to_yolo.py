import os
import cv2
import shutil
import random

random.seed(42)

SOURCE = "dataset/metal_nut"

OUTPUT_IMAGES = "yolo_dataset/images"
OUTPUT_LABELS = "yolo_dataset/labels"

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_IMAGES, split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_LABELS, split), exist_ok=True)


def save_yolo_label(mask_path, label_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        # No defect
        open(label_path, "w").close()
        return

    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    H, W = mask.shape

    x_center = (x + w / 2) / W
    y_center = (y + h / 2) / H
    w = w / W
    h = h / H

    with open(label_path, "w") as f:
        f.write(f"0 {x_center} {y_center} {w} {h}\n")


def split_data(data):
    random.shuffle(data)

    total = len(data)

    train = data[:int(total * 0.7)]
    val = data[int(total * 0.7):int(total * 0.85)]
    test = data[int(total * 0.85):]

    return train, val, test


# -----------------------------
# GOOD IMAGES
# -----------------------------

good_folder = os.path.join(SOURCE, "train", "good")

good_images = [
    os.path.join(good_folder, f)
    for f in os.listdir(good_folder)
    if f.endswith(".png")
]

train, val, test = split_data(good_images)

for split, images in zip(["train", "val", "test"], [train, val, test]):

    for img in images:

        filename = os.path.basename(img)

        shutil.copy(
            img,
            os.path.join(OUTPUT_IMAGES, split, filename)
        )

        open(
            os.path.join(
                OUTPUT_LABELS,
                split,
                filename.replace(".png", ".txt")
            ),
            "w"
        ).close()


# -----------------------------
# DEFECTIVE IMAGES
# -----------------------------

defect_classes = ["bent", "color", "flip", "scratch"]

defect_data = []

for defect in defect_classes:

    image_folder = os.path.join(SOURCE, "test", defect)
    mask_folder = os.path.join(SOURCE, "ground_truth", defect)

    for file in sorted(os.listdir(image_folder)):

        if file.endswith(".png"):

            image_path = os.path.join(image_folder, file)

            mask_path = os.path.join(
                mask_folder,
                file.replace(".png", "_mask.png")
            )

            defect_data.append((image_path, mask_path))


train, val, test = split_data(defect_data)

for split, dataset in zip(["train", "val", "test"], [train, val, test]):

    for image_path, mask_path in dataset:

        filename = os.path.basename(image_path)

        shutil.copy(
            image_path,
            os.path.join(OUTPUT_IMAGES, split, filename)
        )

        save_yolo_label(
            mask_path,
            os.path.join(
                OUTPUT_LABELS,
                split,
                filename.replace(".png", ".txt")
            )
        )

print("YOLO dataset created successfully!")