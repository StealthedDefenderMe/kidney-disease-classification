
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.cnn_classification.entity.config_entity import DataSplitterEntity
from src.cnn_classification import logger


class DataSplitterOD:

    def __init__(self, config: DataSplitterEntity):
        self.config = config

    def initiate_data_splitting(self):

        # Validate split ratios to ensure train, validation and test sum to 100%.
        # check whether our three percentages add up to 100%.
        total_split = (
            self.config.train_split
            + self.config.validation_split
            + self.config.test_split
        )

        if (
            self.config.train_split <= 0
            or self.config.validation_split <= 0
            or self.config.test_split <= 0
            or abs(total_split - 1.0) > 1e-9
        ):
            raise ValueError(
                "Train, validation and test ratios must be positive and sum to 1."
            )

        # Define the class folders available in our original kidney dataset.
        class_names = self.config.class_names

        # Collect all image paths along with their corresponding class names.
        records = []

        # Collect all image paths & visit each folder path one by one
        for class_name in class_names:

            # Locate the current class folder inside the source dataset.
            class_dir = self.config.source_data_path / class_name

            # Stop execution if any expected class folder is missing.
            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            # Find images inside each folder & collect only image files
            image_paths = [
                path for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            # For every image, we remember two things: Where the image is located & Which class it belongs to.
            for image_path in image_paths:
                records.append({
                    "image_path": image_path,
                    "class_name": class_name
                })

        # Ensure that the source dataset contains images before proceeding.
        if not records:
            raise ValueError("No images found in the source dataset.")

        # Convert collected image records into a DataFrame for easier processing. Like into a table
        df = pd.DataFrame(records)
        # Now we have all our images in one table.

        # Split training and temporary data (test & validation)
        # Stratification maintains approximately the same class distribution.
        train_df, temp_df = train_test_split(
            df,
            test_size=self.config.validation_split + self.config.test_split,
            random_state=self.config.seed,
            stratify=df["class_name"]
        )

        # Calculate the test percentage relative to the temporary dataset.
        relative_test_size = (self.config.test_split / (self.config.validation_split + self.config.test_split))

        # Divide the temporary dataset into validation and test datasets.
        # Stratification is applied again to preserve class distribution.
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=self.config.seed,
            stratify=temp_df["class_name"]
        )

        # Keep all three splits together so we can process them uniformly.
        split_data = { "train": train_df, "val": val_df, "test": test_df }

        # Create the physical dataset folders and copy images into their splits.
        manifest_records = []

        for split_name, split_df in split_data.items():

            for _, row in split_df.iterrows():

                # Get the original image path and its destination class folder.
                source_path = Path(row["image_path"])
                destination_dir = (
                    self.config.images_dir / split_name / row["class_name"]
                )

                # Create the destination folder if it does not already exist.
                destination_dir.mkdir(parents=True, exist_ok=True)

                # Preserve the original image filename in the destination folder.
                destination_path = destination_dir / source_path.name

                # Prevent accidental overwriting of an existing destination image.
                if destination_path.exists():
                    raise FileExistsError(
                        f"Destination image already exists: {destination_path}"
                    )

                # Copy the image while preserving its file metadata.
                shutil.copy2(source_path, destination_path)

                # Record the source, destination, class and split for tracking.
                manifest_records.append({
                    "image_path": str(source_path),
                    "destination_path": str(destination_path),
                    "class_name": row["class_name"],
                    "split": split_name
                })

        # Create the manifest's parent directory before saving the CSV file.
        self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the split information so we can track which images belong to each split.
        manifest_df = pd.DataFrame(manifest_records)
        manifest_df.to_csv(self.config.manifest_path, index=False)

        # Log the number of images assigned to each dataset split.
        logger.info(
            f"Dataset splitting completed. "
            f"Train: {len(train_df)}, "
            f"Validation: {len(val_df)}, "
            f"Test: {len(test_df)}"
        )

        # Return the manifest for further inspection or use by other components.
        return manifest_df