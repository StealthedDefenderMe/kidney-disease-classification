
import csv
import hashlib
import math
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import DataTransformationEntity
from src.cnn_classification.utils.image_preprocessing import (
    decode_and_resize_image,
)

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
        self.manifest_path = self.config.root_dir / "split_manifest.csv"

    def prepare_split_manifest(self):

        logger.info("Preparing dataset split manifest...")

        data_path = Path(self.config.data_path)

        # Now we'll collect all image paths and their class labels before splitting them.
        # Collect image paths and class labels
        class_names = sorted(
            [folder.name for folder in data_path.iterdir() if folder.is_dir()]
        )

        print("Class mapping:", dict(enumerate(class_names)))

        train_split = self.config.train_split
        validation_split = self.config.validation_split
        test_split = self.config.test_split

        split_ratios = (train_split, validation_split, test_split)

        if (
            any(ratio <= 0 or ratio >= 1 for ratio in split_ratios)
            or not math.isclose(sum(split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-8)
        ):
            raise ValueError(
                "Training, validation, and test split ratios must be between 0 and 1 "
                "and sum to 1."
            )

        image_records = []

        for label, class_name in enumerate(class_names):
            class_dir = data_path / class_name

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() == ".jpg":

                    image_hash = hashlib.sha256()

                    with image_path.open("rb") as image_file:
                        for chunk in iter(
                            lambda: image_file.read(1024 * 1024), b""
                        ):
                            image_hash.update(chunk)

                    image_records.append({
                        "path": image_path,
                        "label": label,
                        "class_name": class_name,
                        "sha256": image_hash.hexdigest(),
                    })

        records_by_hash = {}

        for record in image_records:
            records_by_hash.setdefault(record["sha256"], []).append(record)

        unique_records = []
        canonical_by_hash = {}

        for image_hash, records in records_by_hash.items():

            if len({record["label"] for record in records}) > 1:
                paths = ", ".join(str(record["path"]) for record in records)

                raise ValueError(
                    "Byte-identical images have conflicting class labels: "
                    f"{paths}"
                )

            canonical_record = min(
                records, key=lambda record: record["path"].as_posix()
            )

            unique_records.append(canonical_record)
            canonical_by_hash[image_hash] = canonical_record

        unique_records.sort(key=lambda record: record["path"].as_posix())

        image_paths = [str(record["path"]) for record in unique_records]
        labels = [record["label"] for record in unique_records]

        if not image_paths:
            raise ValueError(f"No .jpg images found under {data_path}.")

        # Split into training data and a temporary validation/test set.
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths,
            labels,
            test_size=(validation_split + test_split),
            random_state=self.config.seed,
            stratify=labels
            # stratify preserves the proportion of each kidney-disease class across the splits.
            # With stratify=labels, your training, validation, and testing sets will each maintain approximately these same proportions.
        )

        # # Second split: validation and testing
        test_ratio = test_split / (validation_split + test_split)

        # Split the temporary set into validation and test using their relative ratios.
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=test_ratio,
            random_state=self.config.seed,
            stratify=y_temp
            # Without stratify, random splitting could accidentally put too many Tumor images in training and too few in testing.
        )

        split_by_hash = {}

        unique_record_by_path = {
            str(record["path"]): record for record in unique_records
        }

        for paths, split_name in (
            (X_train, "train"),
            (X_val, "validation"),
            (X_test, "test"),
        ):
            for path in paths:
                split_by_hash[unique_record_by_path[path]["sha256"]] = split_name

        self.config.root_dir.mkdir(parents=True, exist_ok=True)

        manifest_fields = [
            "relative_path",
            "class_name",
            "sha256",
            "canonical_path",
            "is_canonical",
            "duplicate_count",
            "split",
            "seed",
            "train_split",
            "validation_split",
            "test_split",
        ]

        with self.manifest_path.open(
            "w", newline="", encoding="utf-8"
        ) as manifest_file:

            writer = csv.DictWriter(
                manifest_file,
                fieldnames=manifest_fields
            )

            writer.writeheader()

            for record in image_records:
                canonical_record = canonical_by_hash[record["sha256"]]

                writer.writerow({
                    "relative_path": record["path"].relative_to(data_path).as_posix(),
                    "class_name": record["class_name"],
                    "sha256": record["sha256"],
                    "canonical_path": canonical_record["path"].relative_to(
                        data_path
                    ).as_posix(),
                    "is_canonical": record["path"] == canonical_record["path"],
                    "duplicate_count": len(records_by_hash[record["sha256"]]),
                    "split": split_by_hash[record["sha256"]],
                    "seed": self.config.seed,
                    "train_split": train_split,
                    "validation_split": validation_split,
                    "test_split": test_split,
                })

        logger.info(
            "Saved split manifest to %s (%d source images, %d unique images).",
            self.manifest_path,
            len(image_records),
            len(unique_records),
        )

    def load_datasets(self, requested_splits):

        logger.info(
            "Loading datasets for splits: %s",
            requested_splits
        )

        data_path = Path(self.config.data_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Split manifest not found: {self.manifest_path}. "
                "Run prepare_split_manifest() first."
            )

        # Read the existing manifest instead of creating new splits.
        with self.manifest_path.open(
            "r", newline="", encoding="utf-8"
        ) as manifest_file:
            records = list(csv.DictReader(manifest_file))

        # Only canonical images are loaded. Duplicate files remain recorded in the manifest.
        records = [
            record for record in records
            if record["is_canonical"].lower() == "true"
            and record["split"] in requested_splits
        ]

        if not records:
            raise ValueError(
                f"No images found for requested splits: {requested_splits}"
            )

        # Maintain the same class-to-label mapping used when the manifest was created.
        class_names = sorted({
            record["class_name"] for record in records
        })

        # Get all class names from the manifest, not just the requested splits.
        with self.manifest_path.open(
            "r", newline="", encoding="utf-8"
        ) as manifest_file:
            all_records = list(csv.DictReader(manifest_file))

        class_names = sorted({
            record["class_name"] for record in all_records
        })

        class_to_label = {
            class_name: label
            for label, class_name in enumerate(class_names)
        }

        # takes one image path and its label, then prepares it for your VGG16 model.
        def load_image(path, label):
            image = decode_and_resize_image(
                path,
                self.config.transformed_image_size,
            )
            return image, label

        # converts your image paths and labels into a TensorFlow dataset that your model can process efficiently.
        def create_dataset(paths, labels, training=False):
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

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

        split_datasets = {}
        split_image_paths = {}

        for split_name in requested_splits:

            split_records = [
                record for record in records
                if record["split"] == split_name
            ]

            if not split_records:
                raise ValueError(f"No images found for split: {split_name}")

            paths = [
                str(data_path / record["relative_path"])
                for record in split_records
            ]

            labels = [
                class_to_label[record["class_name"]]
                for record in split_records
            ]

            # Used to learn model weights; images are shuffled.
            # Used during training to monitor performance and help select the best model.
            # Exact duplicate images are excluded, but patient/scan isolation is not guaranteed.
            dataset = create_dataset(
                paths,
                labels,
                training=(split_name == "train")
            )

            # Data augmentation
            # Applying augmentation after caching
            # It says: Don't always show the training images exactly the same way.
            if split_name == "train" and self.config.augmentation:

                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.1),
                    tf.keras.layers.RandomZoom(0.1),
                ])

                dataset = dataset.map(
                    lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )

            # Apply VGG16 preprocessing after augmentation
            dataset = dataset.map(
                lambda x, y: (
                    tf.keras.applications.vgg16.preprocess_input(x),
                    y
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            split_datasets[split_name] = dataset
            split_image_paths[split_name] = paths

        logger.info("Dataset loading completed successfully.")

        return split_datasets, split_image_paths

    # For Training
    def initiate_data_transformation(self):

        logger.info("Starting data transformation...")

        # Prepare the split manifest only if it does not already exist.
        if not self.manifest_path.exists():
            self.prepare_split_manifest()

        # Training pipeline requests training and validation datasets.
        split_datasets, _ = self.load_datasets(
            ["train", "validation"]
        )

        train_dataset = split_datasets["train"]
        validation_dataset = split_datasets["validation"]

        logger.info("Data transformation completed successfully.")

        return train_dataset, validation_dataset

    # For Evaluation
    def initiate_evaluation_transformation(self):

        logger.info("Starting evaluation data transformation...")

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "Split manifest not found. Run the training pipeline first."
            )

        split_datasets, split_image_paths = self.load_datasets(["test"])

        test_dataset = split_datasets["test"]
        test_image_paths = split_image_paths["test"]

        logger.info("Evaluation data transformation completed.")

        return test_dataset, test_image_paths