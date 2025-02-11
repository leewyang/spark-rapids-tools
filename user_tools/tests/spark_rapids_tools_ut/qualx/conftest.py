import pytest
import sys


@pytest.fixture
def get_test_resources_path():
    # pylint: disable=import-outside-toplevel
    if sys.version_info < (3, 9):
        import importlib_resources
    else:
        import importlib.resources as importlib_resources
    pkg = importlib_resources.files('tests.spark_rapids_tools_ut')
    return pkg / 'resources'
