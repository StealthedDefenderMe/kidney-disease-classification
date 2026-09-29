import os
from box.exceptions import BoxValueError # We'll be using box exception array
import yaml
from cnn_classification import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


# I'm using yaml file in my project. To keep it reading i need below function read_yaml
# I'll provide yaml file path to the function and it will return me content of that yaml file
# ensure_annotations checks the types before running the function. Checks whether the values passed to your function match the types you specified.
# ConfigBox: lets us access dictionary values using dot notation, like d.key instead of d["key"].
# Yaml file always returns a python dictionary
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox: # Config box is return type
    """reads yaml file & returns

    Args: path_to_yaml (str): path like input

    Raises: 
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


# Creates all the directories provided in the list.
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created.
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


# Saves Python dictionary data into a JSON file.
@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Json file saved at: {path}")


# Reads a JSON file and returns its data as a ConfigBox object.
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data
    
    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"Json file loaded successfullt: {path}")
    return ConfigBox(content)


# Saves any Python object into a binary file using joblib.
@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (any): data to be saved as binary
        data (Path): path to binary file
    """

    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


# Loads and returns a Python object from a binary file.
@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data
    
    Args:
        path (Path): path to binary file

    Returns:
        Any: Object stored in that file
    """

    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


# Returns the size of a file in KB.
@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
    
    Args:
        path (Path): path of the file
    
    Returns:
        str: size in KB
    """
    size_in_KB = round(os.path.getsize(path)/1024)
    return f"~ {size_in_KB} KB"


# Decodes a Base64 image string and saves it as an image file.
def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


# Reads an image file and converts it into a Base64 string.
def encodeImageIntoBase64(croppedImagePath):
    with open (croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())