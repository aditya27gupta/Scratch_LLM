import logging
import sys
from logging import StreamHandler
from pathlib import Path
import requests


def load_names_data() -> bytes:
    """Download dataset if not cached, return raw bytes."""
    DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    DATA_FILE = "./data/names.txt"
    file_path = Path(DATA_FILE)
    if not file_path.exists():
        logger.info("Downloading names dataset...")
        response = requests.get(url=DATA_URL, allow_redirects=True, timeout=5)
        response.raise_for_status()
        file_path.write_bytes(response.content)
    return file_path.read_bytes()


def _setup_root_logger() -> None:
    logger = logging.getLogger("scratch_llm")
    logger.setLevel(logging.INFO)
    _handler = StreamHandler(sys.stdout)
    _formatter = logging.Formatter("%(funcName)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


_setup_root_logger()
logger = logging.getLogger("scratch_llm")
