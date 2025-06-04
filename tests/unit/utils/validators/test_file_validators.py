import pytest
from app.utils.validators.file_validators import validate_file_size
from app.core.constants import MAX_FILE_SIZE


def test_validate_file_size():
    # Valid sizes
    assert validate_file_size(0, MAX_FILE_SIZE) is True
    assert validate_file_size(1024, MAX_FILE_SIZE) is True  # 1KB
    assert validate_file_size(1024 * 1024, MAX_FILE_SIZE) is True  # 1MB
    assert validate_file_size(MAX_FILE_SIZE, MAX_FILE_SIZE) is True  # Exact limit

    # Invalid sizes
    assert validate_file_size(MAX_FILE_SIZE + 1, MAX_FILE_SIZE) is False
    assert validate_file_size(10 * MAX_FILE_SIZE, MAX_FILE_SIZE) is False

    # Edge cases
    assert validate_file_size(0, 0) is True
    with pytest.raises(TypeError):
        validate_file_size(None, MAX_FILE_SIZE)
    with pytest.raises(TypeError):
        validate_file_size(1024, None)
