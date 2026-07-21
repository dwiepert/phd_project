import pytest
from pathlib import Path
import random

@pytest.fixture(scope="session")
def data_dir():
    """Provides the absolute path to the data directory."""
    return Path(__file__).resolve().parent / "data"

@pytest.fixture(autouse=True)
def seed_random():
    # Use seed from the command line if provided
    pytest.seed = int(42)
    random.seed(pytest.seed)
    print(f"\nRunning test with seed: {pytest.seed}")