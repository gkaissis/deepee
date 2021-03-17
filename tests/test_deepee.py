from deepee import __version__
import toml
from pathlib import Path


def test_version():
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = toml.loads(open(str(path)).read())
    assert __version__ == pyproject["tool"]["poetry"]["version"]
