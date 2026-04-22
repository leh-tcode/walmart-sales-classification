"""
Centralized logging configuration using Loguru.
All modules import `logger` from here for consistent log formatting.
"""

import sys
from pathlib import Path
from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Console handler — INFO and above
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    colorize=True,
)

# File handler — DEBUG and above, rotating daily
logger.add(
    LOG_DIR / "pipeline.log",
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
)

__all__ = ["logger"]
