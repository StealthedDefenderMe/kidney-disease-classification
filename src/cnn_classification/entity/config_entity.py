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