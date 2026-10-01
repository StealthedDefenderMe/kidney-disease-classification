from pathlib import Path
from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import PrepareBaseModelEntity
import tensorflow as tf


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelEntity):
        self.config = config

    # Now let's load the pretrained model.
    def get_base_model(self):
        logger.info("Loading pretrained base model...")

        # VGG16 → pretrained CNN
        # weights="imagenet" → use already learned weights
        # include_top=False → remove VGG16's original 1000-class classifier
        # input_shape=(224, 224, 3) → our images will be 224×224 RGB
        # VGG16
        # ↓
        # CNN feature extraction layers ✅
        # ↓
        # Original ImageNet classifier ❌
        self.model = tf.keras.applications.VGG16(
            weights=self.config.weights,
            include_top=self.config.include_top,
            input_shape=self.config.image_size
        )
        # self.model contains the actual Keras model.
        # self.config  → configuration
        # self.model   → actual neural network

        self.config.root_dir.mkdir(parents=True, exist_ok=True)

        # save this original pretrained model to the path from your config.
        self.model.save(self.config.base_model_path)
        logger.info("Base model saved successfully.")
        # VGG16 pretrained model
        # ↓
        # self.model
        #         ↓
        # save()
        #         ↓
        # base_model.keras
    

    # A method for the updated model.
    # function takes your pretrained VGG16 and builds your kidney-specific model around it.
    def update_base_model(self):
        logger.info("Updating base model...")

        # Freezing VGG16 before training, so that we can reuse useful image features from ImageNet & only teach new layers
        self.model.trainable = False

        self.model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.GlobalAveragePooling2D(), #Takes all the feature maps produced by VGG16 and converts them into a compact list of numbers.
            tf.keras.layers.Dense(self.config.dense_units, activation="relu"), #Your first custom layer. It learns how to combine those extracted features for your kidney problem.
            tf.keras.layers.Dense(self.config.classes, activation="softmax") #Your final classification layer.
        ])

        self.model.save(self.config.updated_base_model_path) #saves this complete model.
        logger.info("Updated base model saved successfully.")
        # Pretrained VGG16
        #     ↓
        # Extract features
        #     ↓
        # GAP
        #     ↓
        # Your Dense layer
        #     ↓
        # 4-class Softmax
        #     ↓
        # Save updated model


    # we should freeze VGG16 before training
    # It means we temporarily stop changing the pretrained VGG16 layers.
    # VGG16 layers       → 🔒 Frozen
    # Your Dense layers  → 🟢 Trainable, but why?
    # VGG16 already learned useful image features from ImageNet. Initially, we want to reuse those features and only teach the new layers:
    # Later, we can optionally unfreeze some VGG16 layers for fine-tuning.
    # thats why we added self.model.trainable = False before sequential
    # VGG16             → 🔒 Frozen
    # GlobalPooling     → 🟢
    # Dense(128)        → 🟢
    # Dense(4)          → 🟢
    # The new layers will learn during training, while VGG16's pretrained weights stay unchanged.


    # Below adding a method that calls both methods in order
    def prepare_full_model(self):
        self.get_base_model()
        self.update_base_model()