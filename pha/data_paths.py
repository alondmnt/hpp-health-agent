"""Simple multi-modal data path handling.

Clean, explicit API for specifying multiple data files.
No magic, no inference, no complexity.

Usage:
    agent(
        user_prompt="Analyze patterns",
        data_paths={
            "cgm": "data/cgm.csv",
            "diet": "data/meals.csv",
            "activity": "data/exercise.csv"
        }
    )
"""
from typing import Dict
from pathlib import Path


def validate_data_paths(data_paths: Dict[str, str]) -> None:
    """Validate data paths for security.

    Args:
        data_paths: Dictionary of data_type → file_path

    Raises:
        ValueError: If any path fails validation
    """
    if not data_paths:
        raise ValueError("data_paths cannot be empty")

    for data_type, path in data_paths.items():
        # Block path traversal
        if ".." in path:
            raise ValueError(
                f"Path traversal not allowed: {path} (data_type: {data_type})"
            )

        # Block absolute paths to system directories
        if path.startswith("/"):
            system_dirs = ["/etc", "/sys", "/proc", "/root"]
            if any(path.startswith(d) for d in system_dirs):
                raise ValueError(
                    f"System directory access not allowed: {path} (data_type: {data_type})"
                )

        # Block command injection characters
        if any(char in path for char in [";", "|", "&", "`", "$"]):
            raise ValueError(
                f"Invalid characters in path: {path} (data_type: {data_type})"
            )


def get_primary_path(data_paths: Dict[str, str]) -> str:
    """Get primary data path (for legacy tool compatibility).

    Args:
        data_paths: Dictionary of data paths

    Returns:
        CGM path if available, otherwise first path

    Example:
        >>> get_primary_path({"cgm": "cgm.csv", "diet": "meals.csv"})
        'cgm.csv'
    """
    # Prefer CGM (most common)
    if "cgm" in data_paths:
        return data_paths["cgm"]

    # Otherwise return first
    return next(iter(data_paths.values()))
