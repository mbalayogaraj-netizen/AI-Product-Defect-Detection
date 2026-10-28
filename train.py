import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ------------------------
# Paths
# ------------------------
TRAIN_DIR = "cnn_dataset/train"
VAL_DIR = "cnn_dataset/validation"

# ------------------------
# Image settings
# ------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------------------
# Data generators
# ------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ------------------------
# CNN Model
# ------------------------
model = Sequential([
    tf.keras.Input(shape=(224, 224, 3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])

# ------------------------
# Compile Model
# ------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ------------------------
# Train
# ------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15
)

# ------------------------
# Save Model
# ------------------------
os.makedirs("model", exist_ok=True)

model.save("model/defect_model.keras")

print("\nModel saved successfully!")