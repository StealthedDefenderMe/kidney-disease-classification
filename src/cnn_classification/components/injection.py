import os
import zipfile
from pathlib import Path
from src.cnn_classification import logger
import kagglehub
import shutil

class DataInjection:
    def __init__(self, config):
        self.config = config

    def initiate_data_injection(self):
        logger.info("Starting data injection...")

        Path(self.config.root_dir).mkdir(parents=True, exist_ok=True)

         # Download dataset
        download_path = kagglehub.dataset_download(
            self.config.source_url
        )

        # Copy downloaded dataset into artifacts/data
        shutil.copytree(
            download_path,
            self.config.root_dir,
            dirs_exist_ok=True
        )

        logger.info("Dataset downloaded successfully.")