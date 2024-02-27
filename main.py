import os.path
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('handwritten_digit_recognition.model')

image_number = 0

while os.path.isfile((f"prepared_digits/d{image_number}.png")):
    try:
        image = cv2.imread(f"prepared_digits/d{image_number}.png")[:,:,0]
        image = np.invert(np.array([image]))
        prediction = model.predict(image)
        print(f"The prediction is {np.argmax(prediction)}")
        plt.imshow(image[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print("Error: ", e)
    finally:
        image_number += 1