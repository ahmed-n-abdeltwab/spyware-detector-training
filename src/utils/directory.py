import os
from typing import Optional


def ensure_directory(path: str, fallback_to_tmp: bool = True) -> str:
    """
    Safely create directory with permission fallback

    Args:
        path: Desired directory path
        fallback_to_tmp: Whether to fallback to /tmp if permission denied

    Returns:
        Path to created directory (may be different if fallback occurred)
    """
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except PermissionError:
        if not fallback_to_tmp:
            raise

        temp_path = f"/tmp/{path.replace('/', '_')}"
        os.makedirs(temp_path, exist_ok=True)
        print(f"Permission denied, using fallback directory: {temp_path}")
        return temp_path
