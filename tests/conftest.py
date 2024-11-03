import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "scripts"))


@pytest.fixture
def fixture_data_dir() -> Path:
    return Path(__file__).parent / "data"
