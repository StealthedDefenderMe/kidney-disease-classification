from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataInjectionEntity:
    source_url: str
    root_dir: Path

@dataclass
class PrepareBaseModelEntity:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    
    # PrepareBaseModel component needs these values.
    image_size: list
    include_top: bool
    classes: int
    weights: str
    dense_units: int

# Our transformation component needs to know where the dataset is located and where transformation-related artifacts should go.
@dataclass
class DataTransformationEntity:
    root_dir: Path # → where transformation artifacts go
    data_path: Path # → where our downloaded dataset is
    transformed_image_size: list
    batch_size: int
    validation_split: float
    train_split: float
    test_split: float
    seed: int
    augmentation: bool