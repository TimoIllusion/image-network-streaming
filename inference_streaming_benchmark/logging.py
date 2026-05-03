import os
from datetime import datetime

from loguru import logger

_configured = False


def setup_logging(log_dir: str = "logs") -> None:
    """Configure the loguru file sink. Idempotent — safe to call multiple times.

    Callers (server.py, client.py) invoke this explicitly so importing the package
    does not create a `logs/` directory or add a sink as a side effect.
    """
    global _configured
    if _configured:
        return
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.add(f"{log_dir}/{ts}.log")
    _configured = True


__all__ = ["logger", "setup_logging"]
