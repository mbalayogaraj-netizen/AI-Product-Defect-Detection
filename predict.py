import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model/defect_model.keras")

# Image path (change this to your image)
img_path = "cnn_dataset/test/defective/000.png"

# Load image
img = image.load_img(img_path, target_size=(224,224))

# Convert to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

confidence = prediction[0][0]

if confidence > 0.5:
    print("\nPrediction : Defective")
    print(f"Confidence : {confidence*100:.2f}%")
else:
    print("\nPrediction : Good")
    print(f"Confidence : {(1-confidence)*100:.2f}%")