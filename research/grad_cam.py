import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from src.cnn_classification.config.configuration import ConfigurationManager
# This file is used to experiment with Grad-CAM (Gradient-weighted Class Activation Mapping)
# to visualize which regions of kidney images influence the model's predictions
# and investigate misclassified images without modifying or retraining the trained model.

config = ConfigurationManager()
prediction_config = config.get_prediction_config()

# Loading best_model.h5
model = tf.keras.models.load_model(prediction_config.model_path)

print("Model loaded successfully")
# model.summary()

# What? -> Gets the VGG16 model from your complete model.
# Why? -> our complete model contains VGG16, GAP, Dense(128), and Dense(4). 
# We need to access VGG16 specifically because Grad-CAM uses its convolutional layers.
base_model = model.get_layer("vgg16")

# What?-> Finds the block5_conv3 layer inside VGG16. This fetches one specific CN layer from VGG16.
# Why?-> It's the final convolutional layer in VGG16. 
# It contains spatial feature maps that Grad-CAM can use to identify which image regions influenced the prediction.
base_model.get_layer("block5_conv3")
print("Grad-CAM target layer found:", "block5_conv3")

# Now next we'll use GRAD-CAM calculation.
# It will use block5_conv3 activations and gradients to create a 
# heatmap showing which regions influenced the predicted class.
# First loading the misclassified csv from dir.
training_config = config.get_training_config()
misclassified_path = training_config.misclassified_predictions_path
all_predictions_path = training_config.all_predictions_path

# Now, We'll read the CSV and select a row where actual_class == "Tumor" and predicted_class == "Cyst".
# misclassified_df = pd.read_csv(misclassified_path)
all_predictions = pd.read_csv(all_predictions_path)

# tumor_as_cyst = misclassified_df[
#     (misclassified_df["actual_class"] == "Tumor") &
#     (misclassified_df["predicted_class"] == "Cyst")
# ]

correct_predictions = all_predictions[
    (all_predictions["actual_class"] == "Tumor") &
    (all_predictions["predicted_class"] == "Tumor") &
    (all_predictions["correct"] == True)
]

# if tumor_as_cyst.empty: raise ValueError("No Tumor → Cyst misclassified images found.")

# selects the first row from the filtered DataFrame.
# selected_image = tumor_as_cyst.iloc[0]
selected_image = correct_predictions.iloc[4]

print("Selected image:", selected_image["image_name"])
print("Actual class:", selected_image["actual_class"])
print("Predicted class:", selected_image["predicted_class"])
print("Confidence:", selected_image["confidence"])

# Next, Find the actual image file and load it with the same preprocessing used during training
data_transform_config = config.get_data_transformation_config()
data_path = Path(data_transform_config.data_path)

image_path = data_path / selected_image["actual_class"] / selected_image["image_name"]

if not image_path.exists(): raise FileNotFoundError(f"Image not found: {image_path}")
print("Image path:", image_path)

# Load and preprocess the image.
# Resizes the image to 224 × 224, matching VGG16's expected input.
image_array = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Converts the image into a numerical array of pixel values so the model can process it.
image_array = tf.keras.utils.img_to_array(image_array)
# Adds a batch dimension. The shape changes from (224, 224, 3) to (1, 224, 224, 3). 
# The 1 means we're processing one image.
image_array = np.expand_dims(image_array, axis=0)
# Adjusts the pixel values into the format VGG16 expects, matching its ImageNet preprocessing.
image_array = tf.keras.applications.vgg16.preprocess_input(image_array)

print("Image prepared:", image_array.shape)

# Next, Get the model's prediction score
# Gets the model's scores for all four classes.
predictions = model.predict(image_array, verbose=0)

# Finds the index of the class with the highest score.
predicted_class_index = np.argmax(predictions[0])

# Converts that score into a percentage.
predicted_confidence = predictions[0][predicted_class_index] * 100

class_names= prediction_config.class_names
# Converts the index into a class name, such as Cyst or Tumor.
print("Model predicted:", class_names[predicted_class_index])
print("Confidence:", round(predicted_confidence, 2), "%")

# Now we'll use Grad-CAM to understand which regions of this kidney image influenced the model's prediction.
# Create the Grad-CAM model
# We're using Grad-CAM to understand why our kidney CNN misclassified a Tumor image as Cyst with 99.56% 
# confidence by highlighting the image regions that influenced its prediction.

# selects VGG16's final convolutional layer.
last_conv_layer = model.get_layer("vgg16").get_layer("block5_conv3")

# creates a model that returns both:
# The feature maps from block5_conv3.
# The final class prediction scores.
base_model = model.get_layer("vgg16")
last_conv_layer = base_model.get_layer("block5_conv3")
grad_model = tf.keras.models.Model(
    inputs=base_model.inputs,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)
print("Grad-CAM model created successfully")

# Next step: Calculate the gradients and generate the heatmap.
# TensorFlow records the operations so it can calculate how the prediction 
# depends on the convolutional features.
with tf.GradientTape() as tape:
    # get both outputs
    # conv_outputs: the image features from the selected convolutional layer.
    # predictions: probabilities for Cyst, Normal, Stone, and Tumor.
    conv_outputs, predictions = grad_model(image_array)

    # Finds the index of the highest probability
    class_index = tf.argmax(predictions[0])
    # Extract the predicted Cyst probability: approximately 0.9956.
    class_score = tf.gather(predictions[0], class_index)

# It measures how much each convolutional feature affects the Cyst score.
grads = tape.gradient(class_score, conv_outputs)
# Average the gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
# Remove the batch dimension
conv_outputs = conv_outputs[0]

# Multiplies each feature channel by its importance weight and combines them into one spatial map.
# This indicates which regions contributed to the selected class score.
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
# Removes dimensions of size 1, leaving a 2D heatmap.
heatmap = tf.squeeze(heatmap)
# Removes negative values so the heatmap
heatmap = tf.maximum(heatmap, 0)
# Scales values approximately between 0 and 1, making the heatmap easier to visualize.
heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
# In simple terms: Grad-CAM finds which convolutional features increased the Cyst score, combines 
# their spatial information, and produces a heatmap showing the influential regions.
print("Grad-CAM heatmap generated:", heatmap.shape)

# Next, we’ll overlay the Grad-CAM heatmap on your original CT image to see which regions influenced the- 
# incorrect Cyst prediction.
# Load original image for visualization
original_image = cv2.imread(str(image_path))
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image = cv2.resize(original_image, (224, 224))

# Resize heatmap to match the image
heatmap_resized = cv2.resize(
    heatmap.numpy(), (224, 224)
)

# Convert heatmap to color
heatmap_uint8 = np.uint8(255 * heatmap_resized)
heatmap_colored = cv2.applyColorMap(
    heatmap_uint8, cv2.COLORMAP_JET
)
heatmap_colored = cv2.cvtColor(
    heatmap_colored, cv2.COLOR_BGR2RGB
)

# Overlay heatmap on original image
overlay = cv2.addWeighted(
    original_image, 0.6,
    heatmap_colored, 0.4, 0
)

# Display results
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original CT Image")
plt.axis("off")

# Grad-CAM heatmap image
plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(f"Overlay: {class_names[predicted_class_index]} Prediction")
plt.axis("off")

plt.tight_layout()
plt.show()