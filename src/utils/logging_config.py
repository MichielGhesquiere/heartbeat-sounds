import logging
import os

_LOGGER_INITIALIZED = False


def setup_logging(level: str = None):
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    level_name = (level or os.getenv("HEART_LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from external libs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    _LOGGER_INITIALIZED = True
    logging.getLogger(__name__).debug("Logging initialized at %s", level_name)
