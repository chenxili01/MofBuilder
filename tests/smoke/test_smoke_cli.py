from pathlib import Path
import sys

import pytest


SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mofbuilder import cli


@pytest.mark.smoke
def test_cli_version(capsys):
    exit_code = cli.main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip()


@pytest.mark.smoke
def test_cli_list_families(capsys):
    exit_code = cli.main(["list-families"])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert "HKUST-1" in lines


@pytest.mark.smoke
def test_cli_list_metals(capsys):
    exit_code = cli.main(["list-metals", "--mof-family", "UIO-66"])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert "Zr" in lines
