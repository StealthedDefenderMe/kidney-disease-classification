# kidney-disease-classification deep learning project

# STEP 1: Create a template.py file first to create folder struture after cloning the repository

# STEP 2: After setting up setup.py create an virtual environment

# STEP 3: Create a virtual environment & activate. Later set setup.py file with requirement.txt

# STEP 4: Setup Logging module & exception handling (In __init__ to automatically trigger it)

# STEP 5: Import it into main.py

# STEP 6: Using python Box package we will do the exception handling inside utils folder
# inside common.py file, I'm keeping all common code i might need thrughout the application

<!-- Workflows -->
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity (Nothing but return type of any function)
5. Update the configuration manager in src config
6. Update the components (Contains data injection, model preparation & evaluation)
7. Update the pipeline (Training as well as prediction pipeline)
8. Update the main.py
9. Update the dvc.yaml (this is gonna track your entire pipeline)
10. app.py (Updating at very last)

## Here we're starting with our first component which is data injection. Here we'll inject the data