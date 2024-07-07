import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = load_model('models/sketch_model.keras')

def convert_to_sketch(image_path):
    userGivenImage = cv2.imread(image_path)
    givenImages_height, givenImages_width = userGivenImage.shape[:2]

    image = load_img(image_path, target_size=(256, 256), color_mode='rgb')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    sketch = model.predict(image)[0]
    sketch = (sketch * 255).astype(np.uint8)

    sketch_r = cv2.resize(sketch, (givenImages_width, givenImages_height))

    sketch_filename = os.path.splitext(os.path.basename(image_path))[0] + '_sketch_by_rifah.png'
    sketch_path = os.path.join('main/static/output_sketches', sketch_filename)
    cv2.imwrite(sketch_path, sketch_r)

    return sketch_path