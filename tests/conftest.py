from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def root_dir():
    return Path("./tests/")


@pytest.fixture(scope="session")
def data_dir():
    return Path("./tests/data/")


@pytest.fixture(scope="session")
def output_dir():
    return Path("./tests/out/")


@pytest.fixture(scope="session")
def size() -> int:
    return 128
