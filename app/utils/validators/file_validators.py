def validate_file_size(size: int, max_size: int) -> bool:
    """
    Validates that the file size does not exceed the limit.

    Args:
        size: File size in bytes
        max_size: Maximum allowed size in bytes

    Returns:
        bool: True if the size is acceptable, False otherwise
    """
    return size <= max_size
