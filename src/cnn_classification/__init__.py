import logging
import os
from datetime import datetime

# There reason we're declaring logging configuration here is because we want to have a single logging configuration for 
# the entire project. If we declare logging configuration in multiple files, it will create multiple log files and it 
# will be difficult to track the logs.
# __init__.py : automatically runs when package loads
# Any other .py → runs only when you execute/import it

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y')}.log" # '%m_%d_%Y_%H_%M_%S' for everytime you run the code

log_path = os.path.join(os.getcwd(), "logs")

os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger("cnnClassificationLogger") # This logger will be used in the entire project. 
# We can use this logger in any file by importing it.