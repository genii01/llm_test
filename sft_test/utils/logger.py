import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=getattr(logging, log_level),
    )
