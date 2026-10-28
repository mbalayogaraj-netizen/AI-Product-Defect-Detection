import os
import shutil
import random

# -----------------------------
# CHANGE THIS PATH IF NEEDED
# -----------------------------
SOURCE = "dataset/metal_nut"

OUTPUT = "cnn_dataset"

random.seed(42)

# Create output folders
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT, split, "good"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, split, "defective"), exist_ok=True)


def split_files(files):
    random.shuffle(files)

    n = len(files)

    train = files[:int(0.7*n)]
    val = files[int(0.7*n):int(0.85*n)]
    test = files[int(0.85*n):]

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

train, val, test = split_files(good_images)

for img in train:
    shutil.copy(img, os.path.join(OUTPUT, "train", "good"))

for img in val:
    shutil.copy(img, os.path.join(OUTPUT, "validation", "good"))

for img in test:
    shutil.copy(img, os.path.join(OUTPUT, "test", "good"))


# -----------------------------
# DEFECTIVE IMAGES
# -----------------------------
test_folder = os.path.join(SOURCE, "test")

defect_types = ["bent", "color", "flip", "scratch"]

defective_images = []

for defect in defect_types:

    folder = os.path.join(test_folder, defect)

    images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".png")
    ]

    defective_images.extend(images)

train, val, test = split_files(defective_images)

for img in train:
    shutil.copy(img, os.path.join(OUTPUT, "train", "defective"))

for img in val:
    shutil.copy(img, os.path.join(OUTPUT, "validation", "defective"))

for img in test:
    shutil.copy(img, os.path.join(OUTPUT, "test", "defective"))

print("Dataset prepared successfully!")