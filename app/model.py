import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = load_model('models/sketch_model.keras')

def convert_to_sketch(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(256, 256), color_mode='rgb')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    # Predict the sketch
    sketch = model.predict(image)[0]
    sketch = (sketch * 255).astype(np.uint8)

    # Save the sketch to the static folder
    sketch_filename = os.path.splitext(os.path.basename(image_path))[0] + '_sketch.png'
    sketch_path = os.path.join('app/static', sketch_filename)
    cv2.imwrite(sketch_path, sketch)

    return sketch_path