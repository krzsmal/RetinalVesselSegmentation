import sys
from pathlib import Path
from loguru import logger
from typing import Optional
from tqdm import tqdm

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """Setup centralized logger configuration."""

    # Remove default handler
    logger.remove()

    # Console handler
    if enable_console:
        logger.add(
            lambda msg: tqdm.write(msg, end=""), 
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )

def get_logger(name: str = None):
    """Get logger instance."""

    if name:
        return logger.bind(name=name)
    return logger

# Initialize logger on import
setup_logger()