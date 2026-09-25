import logging
from enum import IntEnum


class Level(IntEnum):
    """
    Logging levels matching Python's standard-library `logging` 
    values. Because these values intentionally match `logging`,
    a `Level` can be passed anywhere a normal logging level is
    expected.
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
