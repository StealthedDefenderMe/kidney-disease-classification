# This component handles ======> 
# Creating the evaluation directories using your existing create_directories() function.
# Identifying misclassified test images.
# Copying those images into misclassified_images/.
# Generating misclassified_predictions.csv with actual class, predicted class, and confidence.

import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from src.cnn_classification.utils.common import create_directories

# y_true	Actual test labels
# y_pred	Predicted class indexes
# confidence	Highest predicted probability (%)
# test_image_paths	Original paths in the exact order of test_dataset
# class_names	["Cyst", "Normal", "Stone", "Tumor"]
# evaluation_dir	Folder for the CSV report
# misclassified_images_dir	Folder for copied images

def save_misclassified_images(
    y_true,
    y_pred,
    confidence,
    test_image_paths,
    class_names,
    evaluation_dir,
    misclassified_images_dir,
    misclassified_predictions_path
):
    # Create evaluation directories
    create_directories([
        evaluation_dir,
        misclassified_images_dir
    ])

    # Find indexes of incorrectly classified images
    # Get the indexes of test images where actual and predicted labels differ.
    misclassified_indices = np.where(y_true != y_pred)[0]

    results = []

    # Process each misclassified image
    # Iterates over every incorrectly classified image's index.
    for index in misclassified_indices:
        # Gets the original file path of that misclassified image.
        image_path = Path(test_image_paths[index])

        # Converts the actual class index (e.g., 3) into its name (Tumor).
        actual_class = class_names[y_true[index]]
        # Converts the predicted class index into its name (e.g., Cyst).
        predicted_class = class_names[y_pred[index]]

        # Creates the destination path using the folder and original filename.
        destination = misclassified_images_dir / image_path.name
        # Copy image into the misclassified images folder
        shutil.copy2(image_path, destination)

        # Store prediction details
        # Adds one dictionary containing the image's filename, actual class, predicted class, and 
        # confidence to the results list. We'll use this list to create the CSV report.
        results.append({
            "image_name": image_path.name,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "confidence": round(float(confidence[index]), 2)
        })

    # Generate CSV report
    report_path = misclassified_predictions_path

    pd.DataFrame(
        results,
        columns=[
            "image_name",
            "actual_class",
            "predicted_class",
            "confidence"
        ]
    ).to_csv(report_path, index=False)