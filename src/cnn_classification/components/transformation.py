from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import DataTransformationEntity
import tensorflow as tf
# automatically reads images from your folders and creates a TensorFlow dataset.
# It also handles: Loading images, Resizing, Batching, Creating labels
# more modern TensorFlow/Keras approach & returns a tf.data.Dataset, which integrates nicely with: model.fit()

# we implement the actual image transformation.
# we're going to use Keras' ImageDataGenerator.
# Why?:
# It will handle the repetitive work of:
# Reading images from folders
# Resizing them to 224 × 224
# Creating batches of 16
# Applying augmentation
# Converting class folders into labels

class DataTransformation:
    def __init__(self, config: DataTransformationEntity):
        self.config = config

    def initiate_data_transformation(self):

        logger.info("Starting data transformation...")

        data_path = self.config.data_path

        # Training data is the data the model actually learns from.
        # The model looks at these images and their correct labels & adjusts its weights to get better at classification.
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            data_path, # → Where your Normal, Cyst, Tumor, Stone folders are.
            image_size=tuple(self.config.transformed_image_size), # → Every image is resized to 224 × 224.
            batch_size=self.config.batch_size, # → Model receives 16 images at a time.
            validation_split=self.config.validation_split, # Keep 20% of images for validation, leaving 80% for training.
            subset="training", # → This call gets the 80% training portion.
            seed=self.config.seed # → Makes the split consistent/reproducible each time you run it.
        )

        # Validation data is data the model does NOT learn from. Instead, after training on the 800 images, we ask:
        # Okay, model, how well can you classify these 200 images you've never trained on?
        # This helps us monitor whether the model is learning properly.
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            data_path,
            image_size=tuple(self.config.transformed_image_size),
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            subset="validation",
            seed=self.config.seed # seed is configurable. It is not a fixed number. It controls how the dataset is randomly split/shuffled.
        )

        # Data augmentation
        if self.config.augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])

            train_dataset = train_dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y)
            )


        # VGG16 preprocessing
        train_dataset = train_dataset.map(
            lambda x, y: (
                tf.keras.applications.vgg16.preprocess_input(x),
                y
            )
        )

        validation_dataset = validation_dataset.map(
            lambda x, y: (
                tf.keras.applications.vgg16.preprocess_input(x),
                y
            )
        )

        # Performance optimization
        train_dataset = train_dataset.cache().prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        validation_dataset = validation_dataset.cache().prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        logger.info("Data transformation completed successfully.")

        from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import DataTransformationEntity
import tensorflow as tf
# automatically reads images from your folders and creates a TensorFlow dataset.
# It also handles: Loading images, Resizing, Batching, Creating labels
# more modern TensorFlow/Keras approach & returns a tf.data.Dataset, which integrates nicely with: model.fit()

# we implement the actual image transformation.
# we're going to use Keras' ImageDataGenerator.
# Why?:
# It will handle the repetitive work of:
# Reading images from folders
# Resizing them to 224 × 224
# Creating batches of 16
# Applying augmentation
# Converting class folders into labels

class DataTransformation:
    def __init__(self, config: DataTransformationEntity):
        self.config = config

    def initiate_data_transformation(self):

        logger.info("Starting data transformation...")

        data_path = self.config.data_path

        # Training data is the data the model actually learns from.
        # The model looks at these images and their correct labels & adjusts its weights to get better at classification.
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            data_path, # → Where your Normal, Cyst, Tumor, Stone folders are.
            image_size=tuple(self.config.transformed_image_size), # → Every image is resized to 224 × 224.
            batch_size=self.config.batch_size, # → Model receives 16 images at a time.
            validation_split=self.config.validation_split, # Keep 20% of images for validation, leaving 80% for training.
            subset="training", # → This call gets the 80% training portion.
            seed=self.config.seed # → Makes the split consistent/reproducible each time you run it.
        )

        # Validation data is data the model does NOT learn from. Instead, after training on the 800 images, we ask:
        # Okay, model, how well can you classify these 200 images you've never trained on?
        # This helps us monitor whether the model is learning properly.
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            data_path,
            image_size=tuple(self.config.transformed_image_size),
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            subset="validation",
            seed=self.config.seed # seed is configurable. It is not a fixed number. It controls how the dataset is randomly split/shuffled.
        )

        # Data augmentation
        # It says: Don't always show the training images exactly the same way."
        if self.config.augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"), # Sometimes flips the image horizontally.
                tf.keras.layers.RandomRotation(0.1), # Randomly rotates the image slightly.
                tf.keras.layers.RandomZoom(0.1), # Randomly zooms in/out slightly.
            ])

            # Take every training batch (x) and its label (y), modify the image x, but keep the same label y.
            train_dataset = train_dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y)
            )
            # Original: Kidney Tumor → 🫘
            # Augmented: Flipped + rotated → 🫘
            # Label: Tumor → Tumor
            # The image changes, but its class does not change. Thats the whole purpose.


        # VGG16 preprocessing
        # Here we're doing VGG16-specific preprocessing to both datasets.
        train_dataset = train_dataset.map(
            lambda x, y: (
                tf.keras.applications.vgg16.preprocess_input(x), # Takes the images and converts their pixel values into the format VGG16 expects, because VGG16 was originally trained on ImageNet using this preprocessing.
                y
            )
        )

        # NOTE: we don't augment validation images because validation should represent the original/unmodified data.
        # We do the same preprocessing. But no augmentation.
        validation_dataset = validation_dataset.map(
            lambda x, y: (
                tf.keras.applications.vgg16.preprocess_input(x),
                y
            )
        )

        # Why both?:
        # Because VGG16 must receive both training and validation images in the same expected format.

        # Performance optimization
        # This is simply making the dataset faster for training.
        # cache() means: After loading/preparing the data once, keep it in memory so we don't repeatedly read and process it from disk.
        # Without cache:
            # Disk → process → Model
            # Disk → process → Model
            # Disk → process → Model
        # With Cache:
            # Disk → process → Memory
                                # ↓
                              # Model
                              # Model
                              # Model
        train_dataset = train_dataset.cache().prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        # prefetch() means: While the model is training on the current batch, prepare the next batch in advance.
        # Without:
            # Prepare Batch 1 → Train Batch 1
            # Prepare Batch 2 → Train Batch 2

        # With:
            # Prepare Batch 1 → Train Batch 1
                                    #    ↘ Prepare Batch 2
                                    #          ↓
                                    #       Train Batch 2

        # So training doesn't have to sit and wait for the next batch.

        # AUTOTUNE: TensorFlow automatically decides how much data to prepare ahead of time based on your system.
        # Figure out how many batches should be prepared ahead of time so the pipeline runs efficiently.
        # AUTOTUNE = TensorFlow automatically tunes the value for you.

        validation_dataset = validation_dataset.cache().prefetch(
            buffer_size=tf.data.AUTOTUNE
        )
        # For Example: It decides how many batches TensorFlow should prepare ahead of time.
        # refetch:
        #     [Batch 1] → Model training
        #     [Batch 2] → preparing
        #     [Batch 3] → preparing

        logger.info("Data transformation completed successfully.")

        return train_dataset, validation_dataset