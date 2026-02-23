import os
import subprocess
import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (f"{SRC_DIR}:{existing_pythonpath}"
                         if existing_pythonpath else str(SRC_DIR))
    return subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=env,
    )


@pytest.mark.smoke
def test_top_level_import_is_dependency_light():
    result = _run_python(
        "import mofbuilder, sys\n"
        "assert mofbuilder.__version__\n"
        "assert 'mofbuilder.core' not in sys.modules\n"
        "assert 'mofbuilder.md' not in sys.modules\n"
        "print('ok')\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.smoke
def test_submodule_access_is_lazy_and_on_demand():
    result = _run_python(
        "import mofbuilder, sys\n"
        "assert 'mofbuilder.io' not in sys.modules\n"
        "_ = mofbuilder.io\n"
        "assert 'mofbuilder.io' in sys.modules\n"
        "print('ok')\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
