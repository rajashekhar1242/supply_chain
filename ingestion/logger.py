# ingestion/logger.py
import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "ingestion.log")

logging.basicConfig(
    level=logging.INFO,   # instead of DEBUG
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("ingestion.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ingestion")