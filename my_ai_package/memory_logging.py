"""
Memory Logging - Local logging for memory system diagnostics.

All logs stored locally at ~/.myai/logs/memory.log with rotation.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Log directory
LOG_DIR = Path.home() / ".myai" / "logs"
LOG_FILE = LOG_DIR / "memory.log"

# Create logger
_logger = None


def get_logger() -> logging.Logger:
    """Get or create the memory system logger."""
    global _logger

    if _logger is not None:
        return _logger

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    _logger = logging.getLogger("memory")
    _logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if _logger.handlers:
        return _logger

    # File handler with rotation (5MB max, keep 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # Format: timestamp [level] message
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)

    return _logger


# Convenience functions
def log_info(msg: str) -> None:
    """Log info message."""
    get_logger().info(msg)


def log_error(msg: str) -> None:
    """Log error message."""
    get_logger().error(msg)


def log_warning(msg: str) -> None:
    """Log warning message."""
    get_logger().warning(msg)


def log_debug(msg: str) -> None:
    """Log debug message."""
    get_logger().debug(msg)
