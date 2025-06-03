from app.utils.validators.url_validators import validate_url_scheme


def test_validate_url_scheme():
    # Valid HTTPS URLs
    assert validate_url_scheme("https://example.com") is True
    assert validate_url_scheme("https://api.example.org/image.jpg") is True
    assert validate_url_scheme("https://sub.domain.com:8080/path?query=1") is True

    # Invalid URLs (non-HTTPS)
    assert validate_url_scheme("http://example.com") is False
    assert validate_url_scheme("ftp://example.com/file.jpg") is False
    assert validate_url_scheme("file:///local/path/image.jpg") is False

    # Edge cases
    assert validate_url_scheme("https:example.com") is False
