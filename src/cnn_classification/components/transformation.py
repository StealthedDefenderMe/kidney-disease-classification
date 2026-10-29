from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import DataTransformationEntity
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
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

        data_path = self.config.data_path # Where the data is located

        # Now we'll collect all image paths and their class labels before splitting them.
        # Collect image paths and class labels
        class_names = sorted(
            [folder.name for folder in Path(data_path).iterdir() if folder.is_dir()]
        )

        print("Class mapping:", dict(enumerate(class_names)))

        # print("Training class order:", class_names)

        image_paths = []
        labels = []

        for label, class_name in enumerate(class_names):
            class_dir = Path(data_path) / class_name

            # It reads your four class folders and assigns each image a numeric label.
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() == ".jpg":
                    image_paths.append(str(image_path))
                    labels.append(label)

        # Read split ratios from configuration
        train_split = self.config.train_split
        validation_split = self.config.validation_split
        test_split = self.config.test_split

        # Spliting data into training (70%) and temporary (30%)
        # First split: 70% training, 30% temporary.
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths,
            labels,
            test_size=(validation_split + test_split),
            random_state=self.config.seed,
            stratify=labels # stratify preserves the proportion of each kidney-disease class across the splits.
            # With stratify=labels, your training, validation, and testing sets will each maintain approximately these same proportions.
        )

        # # Second split: validation and testing
        test_ratio = test_split / (validation_split + test_split)

        # Spliting temporary data into validation (15%) and testing (15%)
        # Second split: divides that remaining 30% equally into 15% validation and 15% testing.
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=test_ratio,
            random_state=self.config.seed,
            stratify=y_temp
            # Without stratify, random splitting could accidentally put too many Tumor images in training and too few in testing.
        )

        # takes one image path and its label, then prepares it for your VGG16 model.
        def load_image(path, label):
            image = tf.io.read_file(path)  # 1. Read image file
            image = tf.image.decode_jpeg(image, channels=3)  # 2. Decode JPG into RGB
            image = tf.image.resize(
                image, tuple(self.config.transformed_image_size[:2])
            )  # 3. Resize to 224 × 224
            # image = tf.keras.applications.vgg16.preprocess_input(image)  # 4. Normalize for VGG16
            # preprocess_input() applies VGG16's required pixel transformation; it isn't standard 0–1 normalization.
            return image, label  # 5. Return processed image and its label

        # converts your image paths and labels into a TensorFlow dataset that your model can process efficiently.
        def create_dataset(paths, labels, training=False):
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels)) # Create a dataset of paths and labels

            # Shuffle training data
            # Randomizes the order of training images so the model doesn't learn from a fixed sequence. 
            # Validation and test data aren't shuffled here.
            if training:
                dataset = dataset.shuffle(
                    buffer_size=len(paths),
                    seed=self.config.seed
                )

            # Load and preprocess images
            # Calls load_image() for each image to read, resize, and preprocess it. 
            # AUTOTUNE lets TensorFlow optimize parallel processing.
            dataset = dataset.map(
                load_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            # Group images into batches
            # If your batch size is 16, the model processes 16 images per training step instead of all images at once.
            # check params.yaml
            dataset = dataset.batch(self.config.batch_size)
            # Prefetch the next batch
            # Prepares upcoming batches while the model processes the current batch, improving efficiency.
            return dataset.prefetch(tf.data.AUTOTUNE)

        # Used to learn model weights; images are shuffled.
        train_dataset = create_dataset(X_train, y_train, training=True)
        # Used during training to monitor performance and help select the best model.
        validation_dataset = create_dataset(X_val, y_val)
        # Used after training to evaluate performance on unseen images.
        test_dataset = create_dataset(X_test, y_test)

        self.config.root_dir.mkdir(parents=True, exist_ok=True)

        # Data augmentation
        # Applying augmnetation after caching
        # It says: Don't always show the training images exactly the same way."
        if self.config.augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])

            train_dataset = train_dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Apply VGG16 preprocessing after augmentation
        train_dataset = train_dataset.map(
            lambda x, y: (
                tf.keras.applications.vgg16.preprocess_input(x),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        validation_dataset = validation_dataset.map(
            lambda x, y: (
                tf.keras.applications.vgg16.preprocess_input(x),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        test_dataset = test_dataset.map(
            lambda x, y: (
                tf.keras.applications.vgg16.preprocess_input(x),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
            # Original: Kidney Tumor → 🫘
            # Augmented: Flipped + rotated → 🫘
            # Label: Tumor → Tumor
            # The image changes, but its class does not change. Thats the whole purpose.

        logger.info("Data transformation completed successfully.")

        return train_dataset, validation_dataset, test_dataset, X_test