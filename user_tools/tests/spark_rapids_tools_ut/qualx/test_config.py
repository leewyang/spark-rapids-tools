import pytest
from spark_rapids_tools.tools.qualx.config import (
    get_cache_dir,
    get_label,
)


def test_get_cache_dir(monkeypatch):
    # Test with mock environment variable
    monkeypatch.setenv('QUALX_CACHE_DIR', 'test_cache')
    assert get_cache_dir() == 'test_cache'

    # Test without environment variable (should use default)
    monkeypatch.delenv('QUALX_CACHE_DIR')
    assert get_cache_dir() == 'qualx_cache'


def test_get_label(monkeypatch):
    # Test with duration_sum
    monkeypatch.setenv('QUALX_LABEL', 'duration_sum')
    assert get_label() == 'duration_sum'

    # Test with unsupported label
    with pytest.raises(AssertionError):
        monkeypatch.setenv('QUALX_LABEL', 'duration')
        get_label()

    # Test without environment variable (should use default)
    monkeypatch.delenv('QUALX_LABEL')
    assert get_label() == 'Duration'
