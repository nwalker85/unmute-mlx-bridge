"""Portable smoke tests — run on any platform, no model weights required."""

from pytest import CaptureFixture

from unmute_mlx_bridge import main


def test_main_exists() -> None:
    assert callable(main)


def test_main_runs(capsys: CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert "unmute-mlx-bridge" in captured.out
