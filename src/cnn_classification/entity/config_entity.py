from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataInjectionEntity:
    source_url: str
    root_dir: Path