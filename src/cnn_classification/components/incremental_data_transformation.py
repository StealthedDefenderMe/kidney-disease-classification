import tensorflow as tf
import csv
from src.cnn_classification import logger
from src.cnn_classification.utils.image_preprocessing import (
    decode_and_resize_image,
)


class IncrementalDataTransformation:

    def __init__(self, config):
        self.config = config

    # Loads & resizes an image using the existing preprocessing logic
    def load_image(self, image_path, label):
        # This prepares one image in the correct size while keeping its original label.
        image = decode_and_resize_image(
            image_path,
            self.config.transformed_image_size
        )

        return image, label

    # Creates a tensorflow dataset from image paths & their labels
    def create_dataset(self, image_paths, labels, training=False):

        # Create a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, labels)
        )

        # Shuffle only training data, so the model doesn't always see them in the same order.
        if training:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        # Remember our previous load_image() function? This calls it for every image.
        # AUTOTUNE lets TensorFlow decide how many images to process in parallel.
        # how many images TensorFlow processes simultaneously while preparing the data.
        dataset = dataset.map(
            self.load_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Example: You have 100 images and a batch size of 16.
        # - Batch: The model receives 16 images together to learn from them.
        # - Parallel: TensorFlow can decode and resize multiple images at the same time to prepare those 16 images faster.
        # If your batch size is 16, TensorFlow gives the model 16 images at a time instead of one image at a time.
        # how many images the model receives together.
        # Batch is about feeding the model. Parallel is about preparing the food faster.
        dataset = dataset.batch(self.config.batch_size)

        # While the model is learning from the current batch, TensorFlow prepares the next batch in advance.
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    # Applies augmentation to training images and VGG16 preprocessing
    def preprocess_dataset(self, dataset, training=False):

        # Apply augmentation only to training data
        if training and self.config.augmentation:
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])

            dataset = dataset.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Apply VGG16 preprocessing to all images
        # prepares images in the format VGG16 expects.
        dataset = dataset.map(
            lambda x, y: (
                tf.keras.applications.vgg16.preprocess_input(x),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return dataset

    # Retrieves original training images from the existing split manifest
    def get_original_training_images(self):

        manifest_path = self.config.root_dir / "split_manifest.csv"

        image_paths = []
        labels = []

        # Read the existing manifest
        with open(manifest_path, "r") as file:
            # reads each row.
            reader = csv.DictReader(file)

            for row in reader:

                # Include only original training images
                if row["split"] == "train":

                    image_path = self.config.data_path / row["canonical_path"]

                    # Convert class name into its numeric label
                    class_names = self.config.class_names
                    label = class_names.index(row["class_name"])

                    image_paths.append(str(image_path))
                    labels.append(label)

        return image_paths, labels

    # Retrieves newly collected pseudo-labeled images
    def get_pseudo_labeled_images(self):
        # Why do we do this? During incremental training, we combine these original training images-
        # with our new pseudo-labelled images so the model doesn't forget what it learned earlier.

        image_paths = []
        labels = []

        class_names = self.config.class_names

        # Iterate through each class folder
        for class_name in class_names:

            class_dir = self.config.pseudo_labeled_dir / class_name

            if not class_dir.exists():
                continue

            # Collect supported image files
            for image_path in class_dir.iterdir():

                if image_path.is_file() and image_path.suffix.lower() in [
                    ".jpg", ".jpeg", ".png"
                ]:
                    image_paths.append(str(image_path))
                    labels.append(class_names.index(class_name))

        return image_paths, labels

    # This function combines old training images + new pseudo-labelled images into one training dataset.
    def get_combined_training_data(self):

        original_paths, original_labels = self.get_original_training_images()

        new_paths, new_labels = self.get_pseudo_labeled_images()

        # Combine both datasets
        image_paths = original_paths + new_paths
        labels = original_labels + new_labels

        # Now we have images with their corresponding labels.
        # So the model learns from new images while continuing to remember what it learned from the original-
        # images. This helps reduce catastrophic forgetting.
        logger.info(f"Original training images: {len(original_paths)}")
        logger.info(f"New pseudo-labeled images: {len(new_paths)}")
        logger.info(f"Combined training images: {len(image_paths)}")

        return image_paths, labels

    # Retrieves only original validation images
    def get_original_validation_images(self):
        # We use these same original validation images during incremental training to check whether the-
        # model is improving on data it isn't training on.
        manifest_path = self.config.root_dir / "split_manifest.csv"

        image_paths = []
        labels = []

        class_names = self.config.class_names

        # Read the existing split manifest
        with open(manifest_path, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:

                # Include only original validation images
                if row["split"] == "validation":

                    image_path = self.config.data_path / row["canonical_path"]
                    label = class_names.index(row["class_name"])

                    image_paths.append(str(image_path))
                    labels.append(label)

        return image_paths, labels

    # Creates datasets required for incremental training
    def initiate_incremental_transformation(self):

        # Get original + newly collected training images
        train_paths, train_labels = self.get_combined_training_data()

        # Ensure there are new pseudo-labeled images before incremental training
        # Check that new pseudo-labeled images are available
        new_paths, _ = self.get_pseudo_labeled_images()

        if len(new_paths) < self.config.min_new_images:
            raise ValueError(
                f"Need at least {self.config.min_new_images} new images. "
                f"Currently available: {len(new_paths)}"
            )

        # Get only original validation images
        val_paths, val_labels = self.get_original_validation_images()

        # Create TensorFlow datasets
        train_dataset = self.create_dataset(
            train_paths, train_labels, training=True
        )

        validation_dataset = self.create_dataset(
            val_paths, val_labels, training=False
        )

        # Apply augmentation and VGG16 preprocessing
        train_dataset = self.preprocess_dataset(
            train_dataset, training=True
        )

        validation_dataset = self.preprocess_dataset(
            validation_dataset, training=False
        )

        logger.info("Incremental data transformation completed.")

        return train_dataset, validation_dataset