import logging
import sys
from logging import StreamHandler


def _setup_root_logger() -> None:
    logger = logging.getLogger("scratch_llm")
    logger.setLevel(logging.INFO)
    _handler = StreamHandler(sys.stdout)
    _formatter = logging.Formatter("%(funcName)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


_setup_root_logger()
logger = logging.getLogger("scratch_llm")
