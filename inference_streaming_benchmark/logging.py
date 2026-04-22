import os
from datetime import datetime

from loguru import logger

ts = datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = "logs"

os.makedirs(log_dir, exist_ok=True)

logger.add(f"{log_dir}/{ts}.log")
