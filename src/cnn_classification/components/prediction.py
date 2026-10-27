import tensorflow as tf
import numpy as np
from pathlib import Path
import tensorflow as tf


class Prediction:
    def __init__(self, model_path, class_names, image_size):
        self.model_path = Path(model_path)
        self.model = tf.keras.models.load_model(self.model_path)

        self.class_names = class_names
        self.image_size = tuple(image_size)

    def predict(self, image_path):
        image = tf.keras.utils.load_img(
            image_path,
            target_size=self.image_size
        )

        image_array = tf.keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        image_array = tf.keras.applications.vgg16.preprocess_input(image_array)

        predictions = self.model.predict(image_array, verbose=0)

        print("Class names:", self.class_names)
        print("Raw probabilities:", predictions[0])
        print("Probability sum:", np.sum(predictions[0]))

        predicted_index = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        }