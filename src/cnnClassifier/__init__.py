import os
import sys
import logging
from pathlib import Path

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

ROOT_DIR = Path(__file__).resolve().parents[2]
log_dir = ROOT_DIR / "logs"
log_filepath = log_dir / "running_logs.log"
os.makedirs(log_dir, exist_ok=True)

handlers = [logging.StreamHandler(sys.stdout)]
try:
    handlers.insert(0, logging.FileHandler(log_filepath))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=handlers
)

logger = logging.getLogger("cnnClassifierLogger")
