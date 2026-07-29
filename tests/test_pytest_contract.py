import subprocess
import sys


def test_hardware_tests_not_collected_by_default():
    """Verify that default pytest collection excludes hardware tests.

    Runs `pytest --collect-only` in a subprocess and asserts that no
    collected items reference the `tests/hardware` directory. This test
    is expected to fail on Apple Silicon if hardware tests are not
    properly marker-filtered or excluded from default runs.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    normalized_output = output.replace("\\", "/")
    assert "tests/hardware/" not in normalized_output, (
        f"Default pytest collection incorrectly includes hardware tests:\n{output}"
    )
