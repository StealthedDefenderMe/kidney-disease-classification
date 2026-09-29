import os, logging
from pathlib import Path

# Here you'll generate the folder structure using code instead of creating every folder manually

# Logging string
# Instead of usign print statement everywhere use this log statement

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')

project_name = 'cnn_classification'

# Folder which needs to be created
list_of_files = [
    ".github/workflows/.gitkeep", # file is used to force Git to track and preserve an empty directory.
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "main.py",
    "templates/index.html",
]
# Iterate over the list of files and create directories and files as needed
# It loops through file paths, creates the required directories, and creates an empty file if the file doesn't exist or is empty.
# If the file already exists and has content, it simply logs that the file already exists.
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")