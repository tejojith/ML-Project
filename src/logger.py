import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,

)

if __name__ == "__main__":
    logging.info("Logging has started")
    logging.info(f"Log file created at: {LOG_FILE_PATH}")
    logging.info("This is an info message")
    logging.error("This is an error message")
    logging.warning("This is a warning message")
    logging.debug("This is a debug message")
    logging.critical("This is a critical message")