def validate_url_scheme(url: str) -> bool:
    """
    Validates that a URL uses HTTPS.

    Args:
        url: The URL to validate

    Returns:
        bool: True if the URL uses the HTTPS protocol, False otherwise
    """
    return url and url.startswith("https://")
